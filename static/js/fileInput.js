document.getElementById('uploadForm').addEventListener('submit', function (event) {
    event.preventDefault(); // Prevent form submission

    const fileInput = document.getElementById('file');
    const maxSize = 60 * 1024 * 1024; // 1 MB (1 * 1024 * 1024 bytes)

    if (fileInput.files.length > 0) {
        const fileSize = fileInput.files[0].size;

        if (fileSize > maxSize) {
            alert('File size exceeds the limit (60 MB). Please choose a smaller file.');
        } else {
            event.target.submit();
            console.log('File size is within the limit. Proceed with uploading.');
        }
    } else {
        alert('Please select a file to upload.');
    }
});