// used to allow an external user to download a single file from the frontend
import React from 'react';

const DownloadButton = ({ fileName }) => {
    const handleDownload = () => {
        const fileUrl = `${process.env.PUBLIC_URL}/files/${fileName}`;
        const a = document.createElement('a');
        a.href = fileUrl;
        a.download = fileName; // This attribute will trigger the download
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };

    return (
        <button onClick={handleDownload}>
            Download {fileName}
        </button>
    );
};

export default DownloadButton;
