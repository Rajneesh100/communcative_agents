

replace the open ai api key
create db for storing chat session and conversation data:
```
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  phidata/pgvector:16

```

run these command in docker container [ordes db]
```
 GRANT INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO parser;
```
sample chat example:
```
(venv) rajneesh.kumar@MQTT9DYDT4 ai_chat % python db_agent.py 
Enter your query: show me all the orders                    
INFO     Running: SELECT * FROM orders                                                                                             
┏━ Message ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                                                                                 ┃
┃ show me all the orders                                                                                                          ┃
┃                                                                                                                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━ Response (17.3s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                                                                                 ┃
┃ Running:                                                                                                                        ┃
┃                                                                                                                                 ┃
┃  • transfer_task_to_agent_0(task_description=Retrieve all orders from the orders table., expected_output=A list of all orders   ┃
┃    with their details such as purchase_order_id, order_date, buyer_name, etc., additional_information=)                         ┃
┃                                                                                                                                 ┃
┃ Here are all the orders from the database:                                                                                      ┃
┃                                                                                                                                 ┃
┃  1 Order ID: af5cea8e-86c6-4368-8a32-1a870842e62e                                                                               ┃
┃     • Purchase Order ID: 56732                                                                                                  ┃
┃     • Order Date: 2024-10-14                                                                                                    ┃
┃     • Buyer Name: Sports                                                                                                        ┃
┃     • Buyer Address: 456 New Street, ABC City, XYZ State, India - 123456                                                        ┃
┃     • Supplier Name: None                                                                                                       ┃
┃     • Supplier Address: 123 Whatever Street, Paris, France - 112233                                                             ┃
┃     • Currency: $                                                                                                               ┃
┃     • Tax Amount: 0.00                                                                                                          ┃
┃     • Total Amount: 77569.00                                                                                                    ┃
┃     • Created At: 2025-09-18 15:38:52                                                                                           ┃
┃     • Updated At: 2025-09-22 07:31:10                                                                                           ┃
┃     • Completed: False                                                                                                          ┃
┃  2 Order ID: e62f9c8d-c179-4779-bb82-c0a3dbf68386                                                                               ┃
┃     • Purchase Order ID: 34128                                                                                                  ┃
┃     • Order Date: 2025-10-09                                                                                                    ┃
┃     • Buyer Name: Rare Rabbit                                                                                                   ┃
┃     • Buyer Address: 123 Whatever Street, Paris, France - 112233, 456 New Street, ABC City, XYZ State, India - 123456           ┃
┃     • Supplier Name: None                                                                                                       ┃
┃     • Supplier Address: None                                                                                                    ┃
┃     • Currency: $                                                                                                               ┃
┃     • Tax Amount: 0.00                                                                                                          ┃
┃     • Total Amount: 77569.00                                                                                                    ┃
┃     • Created At: 2025-09-22 05:41:59                                                                                           ┃
┃     • Updated At: 2025-09-22 07:31:10                                                                                           ┃
┃     • Completed: False                                                                                                          ┃
┃  3 Order ID: 98824dd6-1f9b-4b6f-a002-5f70ec8c9c46                                                                               ┃
┃     • Purchase Order ID: 6784                                                                                                   ┃
┃     • Order Date: 2024-11-23                                                                                                    ┃
┃     • Buyer Name: Jacks & Jones                                                                                                 ┃
┃     • Buyer Address: 123 Whatever Street, Paris, France - 112233, 456 New Street, ABC City, XYZ State, India - 123456           ┃
┃     • Supplier Name: None                                                                                                       ┃
┃     • Supplier Address: 123 Whatever Street, Paris, France - 112233, 456 New Street, ABC City, XYZ State, India - 123456        ┃
┃     • Currency: $                                                                                                               ┃
┃     • Tax Amount: 0.00                                                                                                          ┃
┃     • Total Amount: 77569.00                                                                                                    ┃
┃     • Created At: 2025-09-22 06:19:22                                                                                           ┃
┃     • Updated At: 2025-09-22 07:31:10                                                                                           ┃
┃     • Completed: False                                                                                                          ┃
┃                                                                                                                                 ┃
┃ If you need more details or specific information about any order, feel free to ask!                                             ┃
┃                                                                                                                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
Enter your query: show me all black items for this order id : 34128
INFO     Running: SELECT li.item_code, li.description, li.quantity, li.unit_price, li.amount, li.delivery_date                     
         FROM line_items li                                                                                                        
         JOIN orders o ON li.order_id = o.id                                                                                       
         WHERE o.purchase_order_id = '34128' AND li.color = 'black'                                                                
INFO     Running: SELECT * FROM conversation_data WHERE order_id = '34128'                                                         
┏━ Message ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                                                                                 ┃
┃ show me all black items for this order id : 34128                                                                               ┃
┃                                                                                                                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━ Response (48.3s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                                                                                 ┃
┃ Running:                                                                                                                        ┃
┃                                                                                                                                 ┃
┃  • transfer_task_to_agent_0(task_description=Retrieve all black items for the order with purchase_order_id 34128.,              ┃
┃    expected_output=..., additional_information=)                                                                                ┃
┃  • transfer_task_to_agent_1(task_description=Fetch past conversation for the order with order_id 34128 to provide context.,     ┃
┃    expected_output=Past conversation related to order_id 34128., additional_information=)                                       ┃
┃                                                                                                                                 ┃
┃ Here is a list of all black items for the order with purchase_order_id 34128:                                                   ┃
┃                                                                                                                                 ┃
┃   1 Description: 1/1 Sleeve Overshirt                                                                                           ┃
┃      • Quantity: 56                                                                                                             ┃
┃      • Unit Price: 65.95                                                                                                        ┃
┃      • Amount: 197.85                                                                                                           ┃
┃   2 Description: 1/1 Sleeve Overshirt                                                                                           ┃
┃      • Quantity: 56                                                                                                             ┃
┃      • Unit Price: 65.95                                                                                                        ┃
┃      • Amount: 989.25                                                                                                           ┃
┃   3 Description: 1/1 Sleeve Overshirt                                                                                           ┃
┃      • Quantity: 56                                                                                                             ┃
┃      • Unit Price: 65.95                                                                                                        ┃
┃      • Amount: 989.25                                                                                                           ┃
┃   4 Description: 1/1 Sleeve Overshirt                                                                                           ┃
┃      • Quantity: 56                                                                                                             ┃
┃      • Unit Price: 65.95                                                                                                        ┃
┃      • Amount: 791.40                                                                                                           ┃
┃   5 Description: 1/1 Sleeve Overshirt                                                                                           ┃
┃      • Quantity: 56                                                                                                             ┃
┃      • Unit Price: 65.95                                                                                                        ┃
┃      • Amount: 329.75                                                                                                           ┃
┃   6 Description: Round Neck 1/1 Sleeve Pullover                                                                                 ┃
┃      • Quantity: 56                                                                                                             ┃
┃      • Unit Price: 68.00                                                                                                        ┃
┃      • Amount: 204.00                                                                                                           ┃
┃   7 Description: Round Neck 1/1 Sleeve Pullover                                                                                 ┃
┃      • Quantity: 56                                                                                                             ┃
┃      • Unit Price: 68.00                                                                                                        ┃
┃      • Amount: 1020.00                                                                                                          ┃
┃   8 Description: Round Neck 1/1 Sleeve Pullover                                                                                 ┃
┃      • Quantity: 56                                                                                                             ┃
┃      • Unit Price: 68.00                                                                                                        ┃
┃      • Amount: 1020.00                                                                                                          ┃
┃   9 Description: Round Neck 1/1 Sleeve Pullover                                                                                 ┃
┃      • Quantity: 56                                                                                                             ┃
┃      • Unit Price: 68.00                                                                                                        ┃
┃      • Amount: 816.00                                                                                                           ┃
┃  10 Description: Round Neck 1/1 Sleeve Pullover                                                                                 ┃
┃      • Quantity: 56                                                                                                             ┃
┃      • Unit Price: 68.00                                                                                                        ┃
┃      • Amount: 340.00                                                                                                           ┃
┃  11 Description: 1/1 Sleeve Pullover                                                                                            ┃
┃      • Quantity: 56                                                                                                             ┃
┃      • Unit Price: 59.00                                                                                                        ┃
┃      • Amount: 1180.00                                                                                                          ┃
┃  12 Description: 1/1 Sleeve Pullover                                                                                            ┃
┃      • Quantity: 56                                                                                                             ┃
┃      • Unit Price: 59.00                                                                                                        ┃
┃      • Amount: 2065.00                                                                                                          ┃
┃  13 Description: 1/1 Sleeve Pullover                                                                                            ┃
┃      • Quantity: 56                                                                                                             ┃
┃      • Unit Price: 59.00                                                                                                        ┃
┃      • Amount: 2360.00                                                                                                          ┃
┃  14 Description: 1/1 Sleeve Pullover                                                                                            ┃
┃      • Quantity: 56                                                                                                             ┃
┃      • Unit Price: 59.00                                                                                                        ┃
┃      • Amount: 2065.00                                                                                                          ┃
┃  15 Description: 1/1 Sleeve Pullover                                                                                            ┃
┃      • Quantity: 56                                                                                                             ┃
┃      • Unit Price: 59.00                                                                                                        ┃
┃      • Amount: 1180.00                                                                                                          ┃
┃                                                                                                                                 ┃
┃ Please note that the item codes were not provided in the data.                                                                  ┃
┃                                                                                                                                 ┃
┃ There are no past conversations related to this order. If you have any specific questions or need further assistance, feel free ┃
┃ to ask!                                                                                                                         ┃
┃                                                                                                                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
Enter your query: set the count of all these black item to 58 for all item of this order id : 34128
INFO     Running: UPDATE line_items SET quantity = 58 FROM orders WHERE line_items.order_id = orders.id AND                        
         orders.purchase_order_id = '34128' AND line_items.color = 'black'                                                         
┏━ Message ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                                                                                 ┃
┃ set the count of all these black item to 58 for all item of this order id : 34128                                               ┃
┃                                                                                                                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━ Response (12.3s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                                                                                 ┃
┃ Running:                                                                                                                        ┃
┃                                                                                                                                 ┃
┃  • transfer_task_to_agent_0(task_description=Update the quantity of all black items for the order with purchase_order_id 34128  ┃
┃    to 58., expected_output=Confirmation that the quantity of all black items for the specified order has been updated to 58.,   ┃
┃    additional_information=)                                                                                                     ┃
┃                                                                                                                                 ┃
┃ It seems that the database is currently in a read-only mode, and I'm unable to perform the update operation at this time.       ┃
┃ Please check the database settings or try again later when write operations are permitted. If there's anything else I can       ┃
┃ assist you with, feel free to let me know!                                                                                      ┃
┃                                                                                                                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
Enter your query: set the count of all these black item to 58 for all item of this order id : 34128
INFO     Running: UPDATE line_items SET quantity = 58 FROM orders WHERE line_items.order_id = orders.id AND                        
         orders.purchase_order_id = '34128' AND line_items.color = 'black'                                                         
INFO     Running: BEGIN                                                                                                            
INFO     Running: ROLLBACK                                                                                                         
┏━ Message ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                                                                                 ┃
┃ set the count of all these black item to 58 for all item of this order id : 34128                                               ┃
┃                                                                                                                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━ Response (16.2s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                                                                                 ┃
┃ Running:                                                                                                                        ┃
┃                                                                                                                                 ┃
┃  • transfer_task_to_agent_0(task_description=Update the quantity of all black items for the order with purchase_order_id 34128  ┃
┃    to 58., expected_output=Confirmation that the quantity of all black items for the specified order has been updated to 58.,   ┃
┃    additional_information=)                                                                                                     ┃
┃                                                                                                                                 ┃
┃ The quantity of all black items for the order with purchase_order_id 34128 has been successfully updated to 58. If you need     ┃
┃ further assistance, feel free to ask!                                                                                           ┃
┃                                                                                                                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
Enter your query: 

```

run query to validate the changes in db: 

```
SELECT model_id, item_code, description, size, quantity, unit_price, amount, delivery_date FROM line_items WHERE
     order_id = 'e62f9c8d-c179-4779-bb82-c0a3dbf68386' AND color = 'black'; 
```
