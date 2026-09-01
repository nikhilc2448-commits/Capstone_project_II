-- Capstone Project II

-- 1. Join customers and orders

SELECT
    o.OrderID,
    o.OrderDate,
    c.CustomerName,
    c.Region,
    o.Product,
    o.Sales,
    o.Profit
FROM capstone_orders o
JOIN capstone_customers c
ON o.CustomerID = c.CustomerID;


-- 2. Region wise sales summary

SELECT
    c.Region,
    SUM(o.Sales) AS TotalSales,
    SUM(o.Profit) AS TotalProfit,
    COUNT(o.OrderID) AS TotalOrders
FROM capstone_orders o
JOIN capstone_customers c
ON o.CustomerID = c.CustomerID
GROUP BY c.Region
ORDER BY TotalSales DESC;


-- 3. Regions with sales above 10 lakh

SELECT
    c.Region,
    SUM(o.Sales) AS TotalSales
FROM capstone_orders o
JOIN capstone_customers c
ON o.CustomerID = c.CustomerID
GROUP BY c.Region
HAVING SUM(o.Sales) > 1000000;


-- 4. Check profit or loss

SELECT
    OrderID,
    Sales,
    Profit,
    CASE
        WHEN Profit < 0 THEN 'Loss'
        ELSE 'Profit'
    END AS Status
FROM capstone_orders;


-- 5. Orders above average sales

SELECT
    OrderID,
    CustomerID,
    Product,
    Sales
FROM capstone_orders
WHERE Sales >
(
    SELECT AVG(Sales)
    FROM capstone_orders
)
ORDER BY Sales DESC;


-- 6. Customer sales using CTE

WITH CustomerSales AS
(
    SELECT
        CustomerID,
        SUM(Sales) AS TotalSales,
        COUNT(OrderID) AS Orders
    FROM capstone_orders
    GROUP BY CustomerID
)

SELECT *
FROM CustomerSales
ORDER BY TotalSales DESC;


-- 7. Rank customers inside each region

WITH RegionSales AS
(
    SELECT
        c.Region,
        o.CustomerID,
        SUM(o.Sales) AS TotalSales
    FROM capstone_orders o
    JOIN capstone_customers c
    ON o.CustomerID = c.CustomerID
    GROUP BY c.Region, o.CustomerID
)

SELECT
    Region,
    CustomerID,
    TotalSales,
    RANK() OVER
    (
        PARTITION BY Region
        ORDER BY TotalSales DESC
    ) AS RankNo
FROM RegionSales;


-- 8. Find orphan customer IDs

SELECT
    o.OrderID,
    o.CustomerID
FROM capstone_orders o
LEFT JOIN capstone_customers c
ON o.CustomerID = c.CustomerID
WHERE c.CustomerID IS NULL;


-- 9. Product category performance

SELECT
    ProductCategory,
    SUM(Sales) AS TotalSales,
    SUM(Profit) AS TotalProfit
FROM capstone_orders
GROUP BY ProductCategory
ORDER BY TotalProfit DESC;


-- 10. Monthly sales trend

SELECT
    DATE_FORMAT(OrderDate,'%Y-%m') AS Month,
    SUM(Sales) AS TotalSales,
    SUM(Profit) AS TotalProfit
FROM capstone_orders
GROUP BY DATE_FORMAT(OrderDate,'%Y-%m')
ORDER BY Month;


-- 11. Top 10 customers

SELECT
    c.CustomerName,
    SUM(o.Sales) AS TotalSales
FROM capstone_orders o
JOIN capstone_customers c
ON o.CustomerID = c.CustomerID
GROUP BY c.CustomerName
ORDER BY TotalSales DESC
LIMIT 10;


-- 12. Segment wise performance

SELECT
    c.Segment,
    SUM(o.Sales) AS TotalSales,
    SUM(o.Profit) AS TotalProfit
FROM capstone_customers c
JOIN capstone_orders o
ON c.CustomerID = o.CustomerID
GROUP BY c.Segment
ORDER BY TotalSales DESC;


-- 13. Profit margin by region

SELECT
    c.Region,
    ROUND(SUM(o.Profit) / SUM(o.Sales) * 100, 2) AS ProfitMargin
FROM capstone_customers c
JOIN capstone_orders o
ON c.CustomerID = o.CustomerID
GROUP BY c.Region
ORDER BY ProfitMargin DESC;


-- 14. Discount analysis

SELECT
    CASE
        WHEN Discount = 0 THEN 'No Discount'
        WHEN Discount <= 0.10 THEN 'Low'
        WHEN Discount <= 0.25 THEN 'Medium'
        ELSE 'High'
    END AS DiscountLevel,
    COUNT(*) AS Orders,
    ROUND(AVG(Profit),2) AS AvgProfit
FROM capstone_orders
GROUP BY DiscountLevel;


-- 15. Best selling products

SELECT
    Product,
    SUM(Quantity) AS TotalQty,
    SUM(Sales) AS TotalSales
FROM capstone_orders
GROUP BY Product
ORDER BY TotalQty DESC
LIMIT 15;