#!/bin/bash    

main() {
    if [ $# -lt 2 ]; then
      echo "GCP project and service account display names are not given"
      exit 1
    fi

    source ./functions.sh
    project=$1
    account=$2
    check-and-create-service-account $account
    add-permissions-for-service-account $project $account
}

main "$@"
