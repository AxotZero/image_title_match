# install python3.7
apt update
apt -y install software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt -y install python3.7
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

# install pip
apt-get update
apt-get -y install python3-pip
pip3 install --upgrade pip


# install git
apt install git -y


# install requirements
yes | pip3 install -r requirements.txt