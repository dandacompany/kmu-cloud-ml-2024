GET /opensearch_dashboards_sample_data_ecommerce/_search
{
  "query": {
    "match_all": {}
  }
}

GET /opensearch_dashboards_sample_data_ecommerce/_search
{
  "query": {
    "match": {
      "category": "Men's Clothing"
    }
  }
}
