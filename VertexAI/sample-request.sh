#! /bin/bash

config="config.json"
# Exxtract project_id, location, and endpoint name from config.yaml
project_id=$(jq -r '.project_id' $config)
location=$(jq -r '.region' $config)
endpoint_name=$(jq -r '.endpoint_display_name' $config)

echo "Project ID: $project_id"
echo "Location: $location"
echo "Endpoint Name: $endpoint_name"

curl \
    -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json" \
    "https://${location}-aiplatform.googleapis.com/v1/projects/$project_id/locations/$location/endpoints/$endpoint_name:predict" \
    -d '{
        "instances": [
            {
                "text": "Arrested Qaida terrorist an India-born WASHINGTON: Abu Musa al-Hindi, one of the principle terror suspects charged with plotting to attack US financial institutions, has been identified as India-born Dhiren Barot. British police on Tuesday charged Barot, 32, of gathering surveillance plans of ...",
            },
            {
                "text": "DiMarco, Riley Get on Ryder Cup Team (AP) AP - Hal Sutton had a good idea what kind of U.S. team he would take to the Ryder Cup. All that changed in the final round of the PGA Championship.",
            },
            {
                "text": "Art Looks Like Fine Investment for Funds (Reuters) Reuters - Some mutual funds invest in stocks;\\others invest in bonds. Now a new breed of funds is offering\\the chance to own fine art.",
            },
            {
                "text": " #39;One in 12 Emails Infected with Virus #39; The number of attempted attacks by computer viruses rocketed in the first half of the year, according to a report published today. ",
            }
        ]
    }'

# The true labels for the instances are: World, Sports, Business, Sci/Tech
