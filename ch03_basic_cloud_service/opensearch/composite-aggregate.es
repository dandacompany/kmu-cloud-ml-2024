GET /opensearch_dashboards_sample_data_ecommerce/_search
{
  "size": 0,
  "aggs": {
    "sales_by_category_and_manufacturer": {
      "composite": {
        "sources": [
          { "category": { "terms": { "field": "category.keyword" } } },
          { "manufacturer": { "terms": { "field": "manufacturer.keyword" } } }
        ]
      },
      "aggs": {
        "total_sales": {
          "sum": { "field": "taxful_total_price" }
        }
      }
    }
  }
}