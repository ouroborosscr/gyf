sudo docker exec 503037f345f1 mongodump -u "admin" -p "gyf424201" --authenticationDatabase admin --archive --db zeek_analysis --collection 3_1_payload | sudo docker exec -i 503037f345f1 mongorestore -u "admin" -p "gyf424201" --authenticationDatabase admin --archive --nsFrom "zeek_analysis.3_1_payload" --nsTo "zeek_analysis.3_1_payload_backup"


#所有的3_1_payload都要改