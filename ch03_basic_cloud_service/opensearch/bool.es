GET /opensearch_dashboards_sample_data_ecommerce/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "category": "Women's Shoes" } },
        { "range": { "taxful_total_price": { "gte": 150 } } }
      ],
      "filter": [
        { "term": { "customer_gender": "MALE" } }
      ],
      "should": [
        { "match": { "manufacturer": "Elitelligence" } }
      ],
      "must_not": [
        { "term": { "sku": "OUT_OF_STOCK" } }
      ]
    }
  }
}
