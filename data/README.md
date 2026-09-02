# Data

This project uses the Sample Superstore dataset, a publicly available retail dataset containing transactional information from a US-based retail business.

## Data Used

The analysis uses two worksheets from the Sample Superstore Excel workbook:

- **Orders** — transaction-level information including customer, product, sales, profit, discount, shipping and order details.
- **Returns** — identifies orders recorded as returned.

The two tables are integrated using `Order ID` to create a transaction-level return indicator.

Following integration and cleaning, the dataset contains:

- **9,994 transaction records**
- **5,009 unique orders**
- **793 unique customers**

The modelling analysis subsequently aggregates transaction records into customer-level behavioural profiles.

## Expected File Structure

To reproduce the analysis locally, place the Sample Superstore workbook in this directory:

```text
data/
├── README.md
└── Sample - Superstore.xls
