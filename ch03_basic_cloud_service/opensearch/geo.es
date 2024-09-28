GET /opensearch_dashboards_sample_data_ecommerce/_search
{
  "query": {
    "bool": {
      "must": {
        "match_all": {}
      },
      "filter": {
        "geo_distance": {
          "distance": "1.6km",
          "geoip.location": {
            "lat": 40.7128,
            "lon": -74.0060
          }
        }
      }
    }
  }
}
