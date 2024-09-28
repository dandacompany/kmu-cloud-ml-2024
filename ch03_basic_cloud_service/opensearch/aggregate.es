GET /opensearch_dashboards_sample_data_ecommerce/_search
{
  "size": 0,
  "aggs": {
    "category_stats": {
      "terms": {
        "field": "category.keyword",
        "size": 10
      },
      "aggs": {
        "avg_order_amount": {
          "avg": {
            "field": "taxful_total_price"
          }
        }
      }
    }
  }
}
