cp ./raw_data/metadata/phenotypes.txt ./transformed_data/targets
#substitute all tabulations with comma, for later .csv conversion
sed -i 's/\t/,/g' ./transformed_data/targets/phenotypes.txt