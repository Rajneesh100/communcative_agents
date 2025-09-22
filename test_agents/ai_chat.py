# agentic_order_bot.py
import os
import uuid
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, register_uuid
from sentence_transformers import SentenceTransformer

# LLM: phi / ollama usage - we will use a small wrapper to call Ollama via phi if available.
# If you use the phi.Agent system, you can plug these into tools. For simplicity we will use a
# wrapper that call a local model through 'ollama' HTTP api or phi.model.ollama if desired.

# --- CONFIG: change if needed ---
ORDERS_DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "parser",
    "user": "parser",
    "password": "parser123",
}

VECTORD_DB = {
    "host": "localhost",
    "port": 6000,
    "dbname": "ai",
    "user": "ai",
    "password": "ai",
}

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# --- helpers for DB connections ---
def orders_conn():
    return psycopg2.connect(
        host=ORDERS_DB["host"],
        port=ORDERS_DB["port"],
        dbname=ORDERS_DB["dbname"],
        user=ORDERS_DB["user"],
        password=ORDERS_DB["password"],
    )

def vector_conn():
    return psycopg2.connect(
        host=VECTORD_DB["host"],
        port=VECTORD_DB["port"],
        dbname=VECTORD_DB["dbname"],
        user=VECTORD_DB["user"],
        password=VECTORD_DB["password"],
    )

register_uuid()  # helpful for uuid roundtrips

# --- Embedding model load (SentenceTransformer) ---
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def embed_texts(texts: List[str]) -> np.ndarray:
    # returns numpy array shape (n, DIM)
    embs = embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embs

# -------------------------
# D B  A G E N T
# -------------------------
class DBAgent:
    """Simple helper to query/update orders and line_items."""
    def __init__(self):
        pass

    def get_order_by_purchase_id(self, purchase_order_id: str) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM orders WHERE purchase_order_id = %s LIMIT 1"
        with orders_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (purchase_order_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    def get_order_by_id(self, order_id: str) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM orders WHERE id = %s LIMIT 1"
        with orders_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (order_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    def list_orders(self, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        if days_back is None:
            sql = "SELECT * FROM orders ORDER BY order_date DESC LIMIT 100"
            params = ()
        else:
            sql = "SELECT * FROM orders WHERE order_date >= (CURRENT_DATE - INTERVAL '%s day') ORDER BY order_date DESC"
            params = (days_back,)
        with orders_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]

    def get_line_items_for_order(self, order_id: str) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM line_items WHERE order_id = %s ORDER BY created_at"
        with orders_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (order_id,))
                return [dict(r) for r in cur.fetchall()]

    def mark_order_completed(self, order_id: str) -> bool:
        sql = "UPDATE orders SET completed = TRUE, updated_at = CURRENT_TIMESTAMP WHERE id = %s RETURNING id"
        with orders_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (order_id,))
                r = cur.fetchone()
                conn.commit()
                return bool(r)

    def search_orders_by_color(self, color: str) -> List[Dict[str, Any]]:
        # searches line_items.color (case-insensitive) and returns matching distinct orders
        sql = """
            SELECT o.* FROM orders o
            JOIN line_items l ON l.order_id = o.id
            WHERE LOWER(l.color) = LOWER(%s)
            GROUP BY o.id
            ORDER BY o.order_date DESC
        """
        with orders_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (color,))
                return [dict(r) for r in cur.fetchall()]

# -------------------------
# R A G  A G E N T
# -------------------------
class RAGAgent:
    """Store and search conversation embeddings in the pgvector Postgres DB.
       All operations MUST be scoped by order_id.
    """
    def __init__(self):
        pass

    def store_message(self, order_id: str, conversation_id: Optional[str], role: str, text: str):
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        emb = embed_texts([text])[0].astype(np.float32)
        with vector_conn() as conn:
            with conn.cursor() as cur:
                sql = "INSERT INTO conversation_vectors (id, order_id, conversation_id, role, text, embedding) VALUES (gen_random_uuid(), %s, %s, %s, %s, %s)"
                cur.execute(sql, (order_id, conversation_id, role, text, emb.tobytes()))
                conn.commit()
        return conversation_id

    def search_similar(self, order_id: str, query: str, top_k: int = 5):
        """Search vectors but only for rows with this order_id. Returns list of dicts [ {id, conversation_id, role, text, score} ]"""
        q_emb = embed_texts([query])[0].astype(np.float32)
        # We'll use <-> operator (cosine or euclidean depending on setup). Use vector_l2_ops or <-> depending on extension.
        # We'll use a parameterized query and pass the vector as bytes.
        with vector_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Using ivfflat L2 or cosine depends on your index; phidata image typically supports "<->"
                sql = """
                SELECT id, conversation_id, role, text, (embedding <-> %s) as distance
                FROM conversation_vectors
                WHERE order_id = %s
                ORDER BY embedding <-> %s
                LIMIT %s
                """
                cur.execute(sql, (q_emb.tobytes(), order_id, q_emb.tobytes(), top_k))
                rows = cur.fetchall()
                # convert distance -> score (smaller is better); we'll return negative distance as score for ranking
                return [{"id": r["id"], "conversation_id": r["conversation_id"], "role": r["role"], "text": r["text"], "score": float(r["distance"])} for r in rows]

# -------------------------
# P L A N N E R  A G E N T
# -------------------------
class PlannerAgent:
    """High-level decision-maker"""
    def __init__(self, db_agent: DBAgent, rag_agent: RAGAgent):
        self.db = db_agent
        self.rag = rag_agent

    def ensure_order_id(self, user_text: str, provided_order_id: Optional[str]) -> Optional[str]:
        """If the user already provided order_id, use it; else return None to indicate we must ask."""
        if provided_order_id:
            return provided_order_id
        # naive detection: if user mentions purchase_order_id like '56732', let caller prompt
        return None

    def handle_query(self, user_text: str, context: Dict[str, Any]):
        """
        context may contain:
          - order_id (string or None)
          - conversation_id (string or None)
        """
        order_id = context.get("order_id")
        must_have_order = any(tok in user_text.lower() for tok in ["order", "purchase", "po", "order_id", "order id", "purchase_order", "purchase_order_id", "mark completed", "mark order"])
        # If the user asked something that likely needs an order_id, force asking.
        if must_have_order and not order_id:
            return {
                "action": "ask_order_id",
                "text": "Please provide the order's purchase_order_id or the order id (UUID) so I can fetch order-specific data. Example: 'purchase_order_id: 56732' or 'order_id: <uuid>'."
            }

        # simple intent classification (expandable)
        if "show orders" in user_text.lower() or "list orders" in user_text.lower():
            days = None
            if "15 day" in user_text.lower() or "15 days" in user_text.lower():
                days = 15
            elif "month" in user_text.lower():
                days = 30
            rows = self.db.list_orders(days_back=days)
            return {"action":"show_orders", "orders": rows}

        if "color" in user_text.lower():
            # parse color... naive extraction: last word
            words = user_text.split()
            color = words[-1]
            results = self.db.search_orders_by_color(color)
            return {"action":"search_by_color", "orders": results}

        if "mark" in user_text.lower() and "completed" in user_text.lower():
            # require order_id
            if not order_id:
                return {"action": "ask_order_id", "text":"Which order (purchase_order_id or id) should I mark completed?"}
            ok = self.db.mark_order_completed(order_id)
            msg = "Marked completed" if ok else "Failed to mark completed; check order_id"
            # store a RAG note about this update
            cid = context.get("conversation_id")
            text = f"Order {order_id} marked completed by user."
            self.rag.store_message(order_id, cid, "system", text)
            return {"action":"mark_completed", "result": msg}

        if "show details" in user_text.lower() or "show order" in user_text.lower() or "get order" in user_text.lower():
            # fetch order and line items
            if not order_id:
                return {"action":"ask_order_id", "text":"Which order (purchase_order_id or id) do you want details for?"}
            # try to accept purchase_order_id: convert if needed
            # check if provided order_id looks like uuid; otherwise treat it as purchase_order_id and fetch id
            order = None
            if len(order_id) == 36 and "-" in order_id:
                order = self.db.get_order_by_id(order_id)
            else:
                order = self.db.get_order_by_purchase_id(order_id)
            if not order:
                return {"action":"not_found", "text":"Order not found"}
            line_items = self.db.get_line_items_for_order(order["id"])
            # retrieve RAG context for this order
            rag_hits = self.rag.search_similar(order["id"], "recent conversation", top_k=5)
            return {"action":"order_details", "order":order, "line_items":line_items, "rag_hits": rag_hits}

        # fallback: generic conversational RAG search (if order_id provided)
        if order_id:
            # search past conv for this order_id
            hits = self.rag.search_similar(order_id, user_text, top_k=5)
            return {"action":"rag_search", "hits":hits}

        return {"action":"unknown", "text":"Sorry, I couldn't interpret. Examples: 'show orders last 15 day', 'show order 56732 details', 'mark order 56732 completed', 'list orders color black'."}

# -------------------------
# O R C H E S T R A T O R  (CLI)
# -------------------------
def parse_order_id_from_input(text: str) -> Optional[str]:
    # naive extraction: look for patterns like 'purchase_order_id: 56732' or 'order_id: <uuid>'
    if "purchase_order_id" in text:
        parts = text.split("purchase_order_id")
        if len(parts) > 1:
            s = parts[1].strip(" :\t")
            return s.split()[0]
    if "order_id" in text:
        parts = text.split("order_id")
        if len(parts) > 1:
            s = parts[1].strip(" :\t")
            return s.split()[0]
    # fallback: if the text is a 36-char uuid
    tok = text.strip()
    if len(tok) == 36 and "-" in tok:
        return tok
    return None

def run_cli():
    db = DBAgent()
    rag = RAGAgent()
    planner = PlannerAgent(db, rag)
    print("Agentic RAG Order Bot (type 'exit' to quit)\n")
    context = {"order_id": None, "conversation_id": None}

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in ("exit","quit"):
            break

        # quick check if user supplied order_id inline
        maybe = parse_order_id_from_input(user_text)
        if maybe:
            # if this looks like purchase_order_id numeric, try map to internal id
            # attempt to fetch order
            ordrow = db.get_order_by_purchase_id(maybe)
            if ordrow:
                context["order_id"] = ordrow["id"]
                print(f"[context] bound to order_id {context['order_id']} (purchase_order_id {maybe})")
            else:
                # maybe it's UUID
                ordrow = db.get_order_by_id(maybe)
                if ordrow:
                    context["order_id"] = ordrow["id"]
                    print(f"[context] bound to order_id {context['order_id']}")
                else:
                    # store as purchase_order_id raw; planner will ask if needed
                    context["order_id"] = maybe

        # Planner decides action
        resp = planner.handle_query(user_text, context)

        # handle planner responses
        if resp["action"] == "ask_order_id":
            print("Bot:", resp.get("text"))
            # prompt user now synchronously
            order_input = input("Provide purchase_order_id or order_id: ").strip()
            # bind context
            maybe2 = parse_order_id_from_input(order_input) or order_input
            ordrow = db.get_order_by_purchase_id(maybe2)
            if ordrow:
                context["order_id"] = ordrow["id"]
                print(f"[context] bound to order_id {context['order_id']}")
            else:
                ordrow = db.get_order_by_id(maybe2)
                if ordrow:
                    context["order_id"] = ordrow["id"]
                    print(f"[context] bound to order_id {context['order_id']}")
                else:
                    print("Bot: couldn't find that order; please check the id.")
            continue

        if resp["action"] == "show_orders":
            rows = resp.get("orders",[])
            print("Bot: Found %d orders" % len(rows))
            for r in rows[:20]:
                print(f"- purchase_order_id={r['purchase_order_id']} order_date={r['order_date']} total_amount={r.get('total_amount')}")
            continue

        if resp["action"] == "search_by_color":
            rows = resp.get("orders",[])
            print(f"Bot: Found {len(rows)} orders containing that color")
            for r in rows[:20]:
                print(f"- purchase_order_id={r['purchase_order_id']} order_date={r['order_date']} total_amount={r.get('total_amount')}")
            continue

        if resp["action"] == "mark_completed":
            print("Bot:", resp.get("result"))
            continue

        if resp["action"] == "order_details":
            order = resp["order"]
            print("Bot: Order:", order["purchase_order_id"], "date:", order["order_date"], "completed:", order["completed"])
            print("Buyer:", order.get("buyer_name"), order.get("buyer_address"))
            print("Supplier:", order.get("supplier_name"), order.get("supplier_address"))
            print("Line items:")
            for li in resp["line_items"]:
                print(" -", li["model_id"], li["description"], li["color"], li["size"], li["quantity"], li["unit_price"])
            print("RAG hits (recent conv for this order):")
            for h in resp["rag_hits"]:
                print(f" - (score {h['score']:.4f}) {h['role']}: {h['text'][:200]}")
            # also store user query into RAG for future reference
            cid = context.get("conversation_id") or str(uuid.uuid4())
            context["conversation_id"] = cid
            rag.store_message(order["id"], cid, "user", user_text)
            continue

        if resp["action"] == "rag_search":
            hits = resp.get("hits",[])
            print("Bot: Found related conversation snippets for that order:")
            for h in hits:
                print(f" - (dist {h['score']:.4f}) {h['role']}: {h['text'][:200]}")
            # store this user query too
            cid = context.get("conversation_id") or str(uuid.uuid4())
            context["conversation_id"] = cid
            rag.store_message(context["order_id"], cid, "user", user_text)
            continue

        if resp["action"] == "not_found":
            print("Bot:", resp.get("text"))
            continue

        # fallback / unknown
        print("Bot:", resp.get("text", "Sorry I couldn't handle this."))
        # store unknown query to rag if order bound
        if context.get("order_id"):
            cid = context.get("conversation_id") or str(uuid.uuid4())
            context["conversation_id"] = cid
            rag.store_message(context["order_id"], cid, "user", user_text)

if __name__ == "__main__":
    run_cli()
