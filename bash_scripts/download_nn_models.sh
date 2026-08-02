#clone the repository with the models in the nn_trained_models directory
cd pipelines/nn_trained_models
git clone https://github.com/riccardo03-maker/Neural_networks_antimicrobial_resistance

#extract the installed models
mv Neural_networks_antimicrobial_resistance/*fusion* ./
rm -rf Neural_networks_antimicrobial_resistance
cd ../../