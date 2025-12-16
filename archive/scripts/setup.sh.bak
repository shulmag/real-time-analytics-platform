#!/bin/bash    

main() {
    if [ $# -lt 2 ]; then
      echo "GCP service account display name and working directory are not given"
      exit 1
    fi

    source ./functions.sh
    create-creds-file $1 $2
    download-python-packages
}

main "$@"
