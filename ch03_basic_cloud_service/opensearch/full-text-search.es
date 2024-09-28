GET /opensearch_dashboards_sample_data_ecommerce/_search
{
  "query": {
    "match_phrase": {
      "products.product_name": {
        "query": "Chinos",
        "slop": 1
      }
    }
  }
}
