# Pufferfish

1. git clone 
2. cd Pufferfish/
3. pip install -r requirements.txt
4. pyinstaller --noconfirm --onedir --console   --collect-all torch   --hidden-import torch   --add-data "models/policy_network/CN2_BN2_RLROP.pth:models/policy_network"   --add-data "models/policy_network/move_mapping.json:models/policy_network"   --add-data "tablebases/gaviota:tablebases/gaviota"   --add-data "tablebases/polyglot:tablebases/polyglot"   uci_wrapper.py
5. Wyeksportuj silnik z folderu /dist