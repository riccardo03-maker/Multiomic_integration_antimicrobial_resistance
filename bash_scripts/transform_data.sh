#install R and Python requirements
Rscript R_scripts/install_requirements.R
python -m pip install -r requirements.txt

#decompress raw data
unzip ./raw_data/features_gpa_expr_snps.zip -d ./raw_data
unzip ./raw_data/metadata.zip -d ./raw_data

#create folders to store transformed data
mkdir transformed_data
mkdir transformed_data/features
mkdir transformed_data/targets

#create reference list with the 414 strains. This is just the list of strains in snps data, which does not contain extra strains
cp ./raw_data/features_gpa_expr_snps/snps/snps_strains_list.txt ./transformed_data/strains_list.txt

#use the Python script to create sparse matrix of features for the three types of omic data
python functions/data_transformation.py

cp ./raw_data/metadata/phenotypes.txt ./transformed_data/targets
#substitute all tabulations with comma, for later .csv conversion
sed -i 's/\t/,/g' ./transformed_data/targets/phenotypes.txt

#use R scripts to create .csv file of antibiotic resistance classes
Rscript R_scripts/targets_csv.R
