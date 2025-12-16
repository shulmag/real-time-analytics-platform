#!/bin/bash    

# check if a given service account exists
check-service-account() {
    service_account=`gcloud iam service-accounts list | awk -v display_name=$1 '$1 == display_name { print $2 }'`
    if [ ! $service_account ]; then
      echo "-1"
    else
      echo $service_account
    fi
}

# check if a given service account exists; if it doesn't, create one
check-and-create-service-account() {
  r=$(check-service-account $1)  
  if [ "$r" == "-1" ]; then
      echo "The given service account does not exist; creating one"
      `gcloud iam service-accounts create $1 --display-name=$1`
  else
      echo "The given service account already exists in the current project."
  fi
}

# add relevant ficc.ai permissions for an existing service account
# This functions adds relevant policy bindings to the given account (display name) in the given project
add-permissions-for-service-account() {
  declare -a roles=("bigquery.user" "bigquery.dataEditor" "bigquery.jobUser")
  project=$1
  account=$2
  sa=$(check-service-account $account)
  if [ "$sa" == "-1" ]; then
      echo "The given service account does not exist"
      exit 1
  fi
  for role in "${roles[@]}"
    do
      r=`gcloud projects add-iam-policy-binding $project --member=serviceAccount:$sa --role=roles/$role`  
  done
}

# Create a creds file for the specified service account in the specified directory
create-creds-file() {
  display_name=$1
  dir=$2   
  sa=$(check-service-account $display_name)
  if [ "$sa" == "-1" ]; then
      echo "The given service account does not exist"
      exit 1
  fi
  creds=`gcloud iam service-accounts keys create ${dir}/creds.json --iam-account=$sa`
  echo $creds
}

# Check if a table or view name alreddy exists in a BQ dataset. If not exit.
check-bq-table-or-view-name() {
    datasetId=$1
    name=$2

    existing_name=`bq ls $datasetId | awk -v n=$name '$1 == n { print $1 }'`
    if [ $existing_name ]; then
	echo "The given table or view already exists"
	exit 1
    else
	echo "The given table or view does not exist"
    fi
}

# Check if a sql query is valid using the --dry_run bq flag 
validate-sql-query() {
    query_file=$1
    r=`cat $query_file | bq query --dry_run --use_legacy_sql=false`
    echo $r
    if grep -q -i "error" <<< $r; then
	exit 1
    fi
}

create-bq-view() {
    datasetId=$1
    name=$2
    query_file=$3
    query=`cat $query_file`
    r=`bq mk --use_legacy_sql=false --view "$query" $datasetId.$name`
    echo $r
}


download-python-packages() {
  declare -a packages=("pandas" "pytz" "seaborn" "numpy" "scipy" "locale" "tqdm")
  for p in "${packages[@]}"
    do
      python -c "import $p"
      if [ $? -eq "1" ]; then
	echo "installing package " $p
	pip install "$p"
      fi
  done
}
