# Server Deployment and Maintenance Instructions

This document provides a step-by-step guide for deploying the application on a server, including setup, configuration, and maintenance.

## Step 1: Login into server

- Open your terminal, command prompt, or bash shell.
- Run the following command to connect to the server:
  ssh erpsoft@172.16.189.212 -p 2043
- Enter the password when prompted: Erp@soft#
- Upon successful login, you will be in the /erpsoft/home/ directory.

## Step 2: Update server and install necessary tools

- Switch to the superuser (root) to perform administrative tasks:
  su
- Enter the root password: erp@soft
- You are now operating as the root user in the /erpsoft/home/ directory.
- Update the package list to get the latest information on available packages:
  pt update
- Upgrade the installed packages to their latest versions:
  pt upgrade
- Install essential tools for the project:
  pt install python3 python3-venv libaio1 nginx unzip git

## Step 3: Setup Oracle Instant Client

> **Note:** Skip this step if your Oracle database version is newer than 12.0.

1.  Create a directory for the Oracle client:
    sudo mkdir -p /opt/oracle
2.  Navigate into the newly created directory:
    cd /opt/oracle
3.  Download the Oracle Instant Client package:
    sudo wget https://download.oracle.com/otn_software/linux/instantclient/193000/instantclient-basic-linux.x64-19.3.0.0.0dbru.zip
4.  Unzip the downloaded file:
    sudo unzip instantclient-basic-linux.x64-19.3.0.0.0dbru.zip
5.  Remove the zip file to save space:
    sudo rm instantclient-basic-linux.x64-19.3.0.0.0dbru.zip
6.  Configure the dynamic linker to find the Oracle libraries. This command will create a configuration file pointing to the client directory.
    echo /opt/oracle/instantclient_19_3 | sudo tee /etc/ld.so.conf.d/oracle-instantclient.conf
7.  Update the linker cache:
    sudo ldconfig
8.  Open the .bashrc file in a text editor to add environment variables:
    
ano ~/.bashrc
9.  Scroll to the end of the file and add the following lines. These ensure the system can find and use the Oracle client libraries and executables.
    export LD_LIBRARY_PATH=/opt/oracle/instantclient_19_3:
    export PATH=/opt/oracle/instantclient_19_3:
10. Save and exit the editor by pressing Ctrl+X, then Y to confirm, and Enter.
11. Apply the changes to your current session:
    source ~/.bashrc
12. Verify that the installation was successful:
    ldd /opt/oracle/instantclient_19_3/libclntsh.so.19.1
    If the output lists paths for all dependencies without any "not found" messages, the setup is complete. If you see errors, please review the steps.

## Step 4: Setup Git and Clone the Repository

### Part A: Generate an SSH Key on the Server

1.  Generate a new ed25519 SSH key pair. Replace the email with your own.
    ssh-keygen -t ed25519 -C "your_github_email@example.com"
2.  Press Enter to save the key in the default location (~/.ssh/id_ed25519).
3.  Press Enter again when asked for a passphrase to leave it empty.
4.  Press Enter once more to confirm.
5.  Display the public key. The contents of this file need to be added to your GitHub account.
    cat ~/.ssh/id_ed25519.pub
6.  Copy the entire output, which starts with ssh-ed25519.

### Part B: Add the SSH Key to GitHub

1.  In your web browser, log in to your **GitHub** account.
2.  Navigate to **Settings** > **SSH and GPG keys**.
3.  Click **New SSH Key**.
4.  In the **Title** field, provide a descriptive name, such as "Project Ubuntu Server".
5.  In the **Key** field, paste the public key you copied from the server.
6.  Click **Add SSH Key**.

### Part C: Verify the Connection and Clone

1.  Back in the server's terminal, test your SSH connection to GitHub:
    ssh -T git@github.com
2.  If you see a prompt asking to continue connecting, type yes and press Enter.
3.  A successful connection will show a message like: "Hi [Username]! You've successfully authenticated..."
