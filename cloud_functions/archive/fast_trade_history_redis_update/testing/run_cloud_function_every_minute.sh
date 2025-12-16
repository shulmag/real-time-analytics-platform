echo "Make sure the following has been run in a different terminal and credentials are included in the script: $ functions-framework --target hello_http --debug"
for i in $(seq 1 720);
do
    curl localhost:8080
    echo $i
    sleep 100
done