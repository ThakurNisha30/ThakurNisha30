CREATE SCHEMA Shop_Sales;

use Shop_Sales;

CREATE TABLE customers (
    customer_id integer PRIMARY KEY,
    first_name varchar(100),
    last_name varchar(100),
    email varchar(100)
);

CREATE TABLE products (
    product_id integer PRIMARY KEY,
    product_name varchar(100),
    price decimal
);

CREATE TABLE orders (
    order_id integer PRIMARY KEY,
    customer_id integer,
    order_date date
);

CREATE TABLE order_items (
    order_id integer,
    product_id integer,
    quantity integer
);

INSERT INTO customers (customer_id, first_name, last_name, email) VALUES
(1, 'John', 'Doe', 'johndoe@email.com'),
(2, 'Jane', 'Smith', 'janesmith@email.com'),
(3, 'Bob', 'Johnson', 'bobjohnson@email.com'),
(4, 'Alice', 'Brown', 'alicebrown@email.com'),
(5, 'Charlie', 'Davis', 'charliedavis@email.com'),
(6, 'Eva', 'Fisher', 'evafisher@email.com'),
(7, 'George', 'Harris', 'georgeharris@email.com'),
(8, 'Ivy', 'Jones', 'ivyjones@email.com'),
(9, 'Kevin', 'Miller', 'kevinmiller@email.com'),
(10, 'Lily', 'Nelson', 'lilynelson@email.com'),
(11, 'Oliver', 'Patterson', 'oliverpatterson@email.com'),
(12, 'Quinn', 'Roberts', 'quinnroberts@email.com'),
(13, 'Sophia', 'Thomas', 'sophiathomas@email.com');

INSERT INTO products (product_id, product_name, price) VALUES
(1, 'Product A', 10.00),
(2, 'Product B', 15.00),
(3, 'Product C', 20.00),
(4, 'Product D', 25.00),
(5, 'Product E', 30.00),
(6, 'Product F', 35.00),
(7, 'Product G', 40.00),
(8, 'Product H', 45.00),
(9, 'Product I', 50.00),
(10, 'Product J', 55.00),
(11, 'Product K', 60.00),
(12, 'Product L', 65.00),
(13, 'Product M', 70.00);

INSERT INTO orders (order_id, customer_id, order_date) VALUES
(1, 1, '2023-05-01'),
(2, 2, '2023-05-02'),
(3, 3, '2023-05-03'),
(4, 1, '2023-05-04'),
(5, 2, '2023-05-05'),
(6, 3, '2023-05-06'),
(7, 4, '2023-05-07'),
(8, 5, '2023-05-08'),
(9, 6, '2023-05-09'),
(10, 7, '2023-05-10'),
(11, 8, '2023-05-11'),
(12, 9, '2023-05-12'),
(13, 10, '2023-05-13'),
(14, 11, '2023-05-14'),
(15, 12, '2023-05-15'),
(16, 13, '2023-05-16');

INSERT INTO order_items (order_id, product_id, quantity) VALUES
(1, 1, 2),
(1, 2, 1),
(2, 2, 1),
(2, 3, 3),
(3, 1, 1),
(3, 3, 2),
(4, 2, 4),
(4, 3, 1),
(5, 1, 1),
(5, 3, 2),
(6, 2, 3),
(6, 1, 1),
(7, 4, 1),
(7, 5, 2),
(8, 6, 3),
(8, 7, 1),
(9, 8, 2),
(9, 9, 1),
(10, 10, 3),
(10, 11, 2),
(11, 12, 1),
(11, 13, 3),
(12, 4, 2),
(12, 5, 1),
(13, 6, 3),
(13, 7, 2),
(14, 8, 1),
(14, 9, 2),
(15, 10, 3),
(15, 11, 1),
(16, 12, 2),
(16, 13, 3);


select * from customers;
select * from products;
select * from orders;
select * from order_items;

#1)Which product has the highest price? Only return a single row?
use Shop_Sales;
select product_name as product_having_highest_price from products where price in (select max(price) from products);
select product_name from products order by price desc limit 1;

#2)Which customer has made the most orders?
use Shop_Sales;
SELECT customers.customer_id, customers.first_name, customers.last_name, COUNT(orders.order_id) AS total_orders
FROM customers
LEFT JOIN orders
ON customers.customer_id = orders.customer_id
GROUP BY customers.customer_id, customers.first_name, customers.last_name
ORDER BY total_orders DESC;

#3)What’s the total revenue per product?
SELECT products.product_id, product_name, SUM(products.product_id * order_items.quantity) as Revenue
FROM products
LEFT JOIN order_items
ON products.product_id = order_items.product_id
GROUP BY  products.product_id, product_name
ORDER BY Revenue DESC;

#4)Find the day with the highest revenue?
SELECT order_date, sum(price*quantity) AS revenue
FROM orders
left join products
on orders.order_id = products.product_id
left join order_items
on products.product_id = order_items.order_id
group by order_date
order by revenue desc
limit 1;

#5)Find the first order (by date) for each customer.
WITH cte AS (
SELECT
customers.customer_id,
CONCAT(customers.first_name, ' ', customers.last_name) AS full_name,
orders.order_date,
DENSE_RANK() OVER (PARTITION BY customers.customer_id ORDER BY orders.order_date ASC) AS rnk
FROM customers
JOIN orders 
ON orders.customer_id = customers.customer_id
)

SELECT full_name, order_date, customer_id, rnk
FROM cte
WHERE rnk = 1;



#6)Find the top 3 customers who have ordered the most distinct products
SELECT customers.customer_id, customers.first_name, customers.last_name, count(distinct order_items.product_id) as distinct_products
FROM customers
LEFT JOIN orders
ON customers.customer_id = orders.customer_id
LEFT JOIN order_items
ON orders.order_id = order_items.order_id
GROUP BY customers.customer_id, customers.first_name, customers.last_name
ORDER BY distinct_products DESC
LIMIT 3;

#7)Which product has been bought the least in terms of quantity?
SELECT product_name, count(quantity) as total_quantity
FROM products
LEFT JOIN order_items
ON products.product_id = order_items.product_id
group by product_name
order by total_quantity
limit 1;

#8)What is the median order total?
with cte as
(select order_items.order_id, sum(order_items.quantity * products.price) as total_order
from order_items join products 
on products.product_id = order_items.product_id
group by  order_items.order_id)

SELECT total_order AS median_order_total
FROM (
SELECT total_order, ROW_NUMBER() OVER (ORDER BY total_order) AS row_num, COUNT(*) AS total_rows
FROM cte
GROUP BY total_order
) sub
WHERE row_num = CEILING(total_rows / 2.0)

#9)For each order, determine if it was ‘Expensive’ (total over 300), ‘Affordable’ (total over 100), or ‘Cheap’.
WITH cte AS (
SELECT order_id, SUM(quantity * price) AS total_order
FROM order_items
JOIN products ON products.product_id = order_items.product_id
GROUP BY order_id
)
SELECT order_id,
CASE
WHEN total_order > 300 THEN 'Expensive'
WHEN total_order > 100 THEN 'Affordable'
ELSE 'Cheap'
END AS order_category
FROM cte;

#10)Find customers who have ordered the product with the highest price.

WITH cte AS (
SELECT
c.customer_id,
CONCAT(c.first_name, ' ', c.last_name) AS full_name,
p.price,
DENSE_RANK() OVER (ORDER BY p.price DESC) AS rnk
FROM
customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id
ORDER BY p.price DESC
)
SELECT full_name
FROM cte
WHERE rnk = 1;



