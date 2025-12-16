#!/bin/bash    

main() {
    if [ $# -lt 3 ]; then
      echo "Requires three parameters: <datasetId> <view_name> <query_file>"
      exit 1
    fi

    source ./functions.sh
    datasetId=$1
    view_name=$2
    query_file=$3
    check-bq-table-or-view-name $datasetId $view_name
    validate-sql-query $query_file
    create-bq-view $datasetId $view_name $query_file
}

main "$@"
