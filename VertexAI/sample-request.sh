#! /bin/bash
# Classify news headlines with the deployed Vertex AI Endpoint.
#
# Usage:
#   ./sample-request.sh                Classify 4 built-in example headlines
#   ./sample-request.sh --interactive  Type your own headlines, one at a time

config="config.json"
project_id=$(jq -r '.project_id' $config)
location=$(jq -r '.region' $config)
endpoint_name=$(jq -r '.endpoint_display_name' $config)

echo "Project ID: $project_id"
echo "Location: $location"
echo "Endpoint Name: $endpoint_name"

predict_url="https://${location}-aiplatform.googleapis.com/v1/projects/$project_id/locations/$location/endpoints/$endpoint_name:predict"

# Sends one headline to the endpoint and prints the predicted class plus
# every class's probability, sorted highest first.
classify() {
    local headline="$1"
    local payload
    payload=$(jq -n --arg text "$headline" '{instances: [{text: $text}]}')

    local response
    response=$(curl -s \
        -X POST \
        -H "Authorization: Bearer $(gcloud auth print-access-token)" \
        -H "Content-Type: application/json" \
        "$predict_url" \
        -d "$payload")

    local prediction
    prediction=$(echo "$response" | jq -r '.predictions[0]')
    if [[ "$prediction" == "null" || -z "$prediction" ]]; then
        echo "No prediction returned - raw response:"
        echo "$response" | jq . 2>/dev/null || echo "$response"
        return
    fi

    echo "$prediction" | jq -r '
        "  -> \(.predicted_class)\n" +
        (.probabilities | to_entries | sort_by(-.value)
            | map("     \(.key): \((.value * 100 | round))%") | join("\n"))
    '
}

if [[ "$1" == "--interactive" || "$1" == "-i" ]]; then
    echo
    echo "Type a news headline and press Enter to classify it (World / Sports / Business / Sci-Tech)."
    echo "Type 'quit' or press Ctrl+C to stop."
    echo

    while true; do
        read -rp "> " headline
        [[ "$headline" == "quit" || "$headline" == "exit" ]] && break
        [[ -z "$headline" ]] && continue
        classify "$headline"
        echo
    done
else
    # The true labels for these headlines, in order: World, Sports, Business, Sci/Tech
    headlines=(
        "Arrested Qaida terrorist an India-born WASHINGTON: Abu Musa al-Hindi, one of the principle terror suspects charged with plotting to attack US financial institutions, has been identified as India-born Dhiren Barot. British police on Tuesday charged Barot, 32, of gathering surveillance plans of ..."
        "DiMarco, Riley Get on Ryder Cup Team (AP) AP - Hal Sutton had a good idea what kind of U.S. team he would take to the Ryder Cup. All that changed in the final round of the PGA Championship."
        "Art Looks Like Fine Investment for Funds (Reuters) Reuters - Some mutual funds invest in stocks;\others invest in bonds. Now a new breed of funds is offering\the chance to own fine art."
        " #39;One in 12 Emails Infected with Virus #39; The number of attempted attacks by computer viruses rocketed in the first half of the year, according to a report published today. "
    )

    for headline in "${headlines[@]}"; do
        echo
        echo "Headline: $headline"
        classify "$headline"
    done
fi
