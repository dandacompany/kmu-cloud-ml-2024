GET /opensearch_dashboards_sample_data_ecommerce/_search
{
  "query": {
    "match": {
      "day_of_week": "Sunday"
    }
  },
  "highlight": {
    "fields": {
      "day_of_week": {}
    }
  }
}
