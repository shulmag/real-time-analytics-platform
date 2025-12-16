"""

This script connects to an SFTP server and deletes all files that are more than 1 day old. The purpose of this
script is to ensure that the SFTP server directory remains clean and does not accumulate outdated files, which 
could take up unnecessary space and potentially cause issues with storage limits or data management practices.

TODO: Consider enhancing this script to handle exceptions and add logging for better monitoring and debugging.
"""

import paramiko
from datetime import datetime, timedelta

def clean_sftp_old_files(sftp_host: str, sftp_username: str, sftp_password: str, directory: str = '.'):
    """
    Connect to an SFTP server and delete all files that are more than 1 day old.

    Args:
        sftp_host (str): SFTP server hostname.
        sftp_username (str): SFTP server username.
        sftp_password (str): SFTP server password.
        directory (str): Directory to clean up. Default is the root directory.

    Returns:
        str: Confirmation message with the number of files deleted.
    """
    # Establish SFTP connection
    transport = paramiko.Transport((sftp_host, 22))
    transport.connect(username=sftp_username, password=sftp_password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    
    # Get the current time and the cutoff time
    now = datetime.now()
    cutoff_time = now - timedelta(days=7)
    
    # List files in the directory
    files_deleted = 0
    for filename in sftp.listdir_attr(directory):
        file_path = f"{directory}/{filename.filename}"
        file_mod_time = datetime.fromtimestamp(filename.st_mtime)
        if file_mod_time < cutoff_time:
            sftp.remove(file_path)
            files_deleted += 1
            print(f"Deleted file: {file_path}")
    
    # Close the SFTP connection
    sftp.close()
    transport.close()

def main(args):
    """
    Simple entry point to run the clean_sftp_old_files function with hardcoded parameters.
    """
    clean_sftp_old_files('sftp.ficc.ai', 'investortools', 'ipts0520', '/uploads')
    return 'SUCCESS'

if __name__ == "__main__":
    main()
