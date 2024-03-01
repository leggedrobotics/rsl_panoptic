#!/bin/bash

# Assuming HOST_UID and HOST_GID are passed as environment variables
USER_HOME=/home/$HOST_USERNAME

# Check if the user exists; if not, create it with the correct UID and GID
if ! id "$HOST_USERNAME" &>/dev/null; then
    groupadd -g $HOST_GID $HOST_USERNAME
    useradd -u $HOST_UID -g $HOST_GID -d $USER_HOME -s /bin/bash $HOST_USERNAME
    echo "$HOST_USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
fi

export HOME=$USER_HOME
export USER=$HOST_USERNAME
cd $HOME

# Now, switch to the user's environment
sudo -E -u $HOST_USERNAME /bin/bash
