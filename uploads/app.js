const express = require('express');
const { spawn } = require('child_process');
const app = express();
const port = 3000;

app.use(express.static('uploads'));  // Serve static files

app.get('/', (req, res) => {
    res.sendFile(__dirname + '/upload.html');
});

app.get('/predict', (req, res) => {
    const pythonProcess = spawn('python', ['predict.py']);

    pythonProcess.stdout.on('data', (data) => {
        if (!res.headersSent) {
            res.send(`Prediction: ${data}`);
        }
    });

    pythonProcess.stderr.on('data', (data) => {
        if (!res.headersSent) {
            console.error(`Error from Python: ${data}`);
            res.status(500).send(`Error: ${data}`);
        }
    });

    pythonProcess.on('close', (code) => {
        console.log(`Child process exited with code ${code}`);
    });
});

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
