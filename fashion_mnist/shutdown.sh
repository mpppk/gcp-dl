#! /bin/bash

export SLACK_URL=''

curl -X POST --data-urlencode "payload={\"channel\": \"#notification\", \"username\": \"webhookbot\", \"text\": \"dl instance will be shutdown. \", \"icon_emoji\": \":ghost:\"}" $SLACK_URL