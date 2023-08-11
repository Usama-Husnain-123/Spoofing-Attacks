document.addEventListener('DOMContentLoaded', function() {
    const videoStream = document.getElementById('video-stream');
    const captureButton = document.getElementById('capture-button');

    captureButton.addEventListener('click', async function() {
        // Capture the frame from the video stream
        const canvas = document.createElement('canvas');
        canvas.width = videoStream.width;
        canvas.height = videoStream.height;
        const context = canvas.getContext('2d');
        context.drawImage(videoStream, 0, 0, canvas.width, canvas.height);

        // Convert the captured frame to a data URL
        const lastCapturedFrame = canvas.toDataURL('image/jpeg');

        // Send the last captured frame to the server for saving
        const response = await fetch('/capture_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image_data: lastCapturedFrame }),
        });

        if (response.ok) {
            const redirectUrl = '/frame_captured';
            window.location.href = redirectUrl; // Redirect to the frame captured page with image data
        } else {
            alert('Failed to capture frame.');
        }
    });
});
