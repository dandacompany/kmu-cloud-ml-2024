GET /opensearch_dashboards_sample_data_ecommerce/_search
{
  "query": {
    "script": {
      "script": {
        "source": "doc['taxful_total_price'].value > doc['taxless_total_price'].value * 0.9",
        "lang": "painless"
      }
    }
  },
  "_source": ["taxful_total_price", "taxless_total_price"]
}
