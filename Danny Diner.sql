CREATE SCHEMA dannys_diner;

use dannys_diner;
CREATE TABLE sales (customer_id VARCHAR(1),order_date DATE,product_id INTEGER);

INSERT INTO sales(customer_id, order_date, product_id)
VALUES('A', '2021-01-01', '1'),
  ('A', '2021-01-01', '2'),
  ('A', '2021-01-07', '2'),
  ('A', '2021-01-10', '3'),
  ('A', '2021-01-11', '3'),
  ('A', '2021-01-11', '3'),
  ('B', '2021-01-01', '2'),
  ('B', '2021-01-02', '2'),
  ('B', '2021-01-04', '1'),
  ('B', '2021-01-11', '1'),
  ('B', '2021-01-16', '3'),
  ('B', '2021-02-01', '3'),
  ('C', '2021-01-01', '3'),
  ('C', '2021-01-01', '3'),
  ('C', '2021-01-07', '3');
  
  CREATE TABLE menu (
  product_id INTEGER,
  product_name VARCHAR(5),
  price INTEGER
);

INSERT INTO menu
  (product_id, product_name, price)
VALUES
  ('1', 'sushi', '10'),
  ('2', 'curry', '15'),
  ('3', 'ramen', '12');
  
  CREATE TABLE members (
  customer_id VARCHAR(1),
  join_date DATE
);

INSERT INTO members
  (customer_id, join_date)
VALUES
  ('A', '2021-01-07'),
  ('B', '2021-01-09');


##1.What is the total amount each customer spent at the restaurant?
select customer_id, sum(price) as total_amount_spent from sales left join menu on menu.product_id = sales.product_id 
group by customer_id order by total_amount_spent;

##Explaination: sum of price is total amount, which can be taken from menu table and left joining it with sales considering all customers using customer_id.

#2.How many days has each customer visited the restaurant?

select customer_id, count(DISTINCT order_date) as count_customer_visits from sales group by customer_id order by product_name, ;

##Explaination : Customer_id and Join_date are to be important from members table

#3.What was the first item from the menu purchased by each customer?

#4. What is the most purchased item on the menu and how many times was it purchased by all customers?

Select product_name as most_purchased_item, 
			count(sales.product_id) as order_count 
from menu join sales on sales.product_id = menu.product_id 
group by product_name 
order by order_count DESC 
limit 1;

#5.Which item was the most popular for each customer?


