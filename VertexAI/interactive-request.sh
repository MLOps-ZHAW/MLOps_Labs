#! /bin/bash
# Interactively classify your own headlines using the deployed Endpoint.
# Same request/response shape as sample-request.sh, just one headline at a
# time and typed in live instead of hardcoded.

config="config.json"
project_id=$(jq -r '.project_id' $config)
location=$(jq -r '.region' $config)
endpoint_name=$(jq -r '.endpoint_display_name' $config)

echo "Type a news headline and press Enter to classify it (World / Sports / Business / Sci-Tech)."
echo "Type 'quit' or press Ctrl+C to stop."
echo

while true; do
    read -rp "> " headline
    [[ "$headline" == "quit" || "$headline" == "exit" ]] && break
    [[ -z "$headline" ]] && continue

    payload=$(jq -n --arg text "$headline" '{instances: [{text: $text}]}')

    response=$(curl -s -X POST \
        -H "Authorization: Bearer $(gcloud auth print-access-token)" \
        -H "Content-Type: application/json" \
        "https://${location}-aiplatform.googleapis.com/v1/projects/$project_id/locations/$location/endpoints/$endpoint_name:predict" \
        -d "$payload")

    prediction=$(echo "$response" | jq -r '.predictions[0]')
    if [[ "$prediction" == "null" || -z "$prediction" ]]; then
        echo "No prediction returned - raw response:"
        echo "$response" | jq . 2>/dev/null || echo "$response"
        echo
        continue
    fi

    echo "$prediction" | jq -r '
        "  -> \(.predicted_class)\n" +
        (.probabilities | to_entries | sort_by(-.value)
            | map("     \(.key): \((.value * 100 | round))%") | join("\n"))
    '
    echo
done
