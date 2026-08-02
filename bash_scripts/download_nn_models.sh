#clone the repository with the models in the nn_trained_models directory
cd pipelines/nn_trained_models
git clone https://github.com/riccardo03-maker/Neural_networks_antimicrobial_resistance

#extract the installed models
mv Neural_networks_antimicrobial_resistance/early_fusion/* ./early_fusion
mv Neural_networks_antimicrobial_resistance/early_fusion_rf/* ./early_fusion_rf
mv Neural_networks_antimicrobial_resistance/intermediate_fusion/* ./intermediate_fusion
mv Neural_networks_antimicrobial_resistance/intermediate_fusion_rf/* ./intermediate_fusion_rf
mv Neural_networks_antimicrobial_resistance/late_fusion/* ./late_fusion
mv Neural_networks_antimicrobial_resistance/late_fusion_rf/* ./late_fusion_rf

rm -rf Neural_networks_antimicrobial_resistance
cd ../../