const express = require('express');
const bodyParser = require('body-parser');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3000;

// Ensure the uploads directory exists
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir);
}

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Serve static files from the 'uploads' directory
app.use('/uploads', express.static(uploadsDir));

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});

const upload = multer({ storage: storage });

// Serve the HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'upload.html'));
});

// Endpoint to handle image uploads and predictions
app.post('/predict', upload.single('image'), (req, res) => {
    const imagePath = req.file.path;

    // Log the file path
    console.log(`Uploaded file path: ${imagePath}`);

    // Use the full path to the Python executable
    const pythonExecutable = '/usr/bin/python3'; // Replace with the correct path
    const pythonProcess = spawn(pythonExecutable, ['predict.py', imagePath]);

    let prediction = '';
    let errorMessages = '';
    let errorOccurred = false;

    pythonProcess.stdout.on('data', (data) => {
        prediction += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
        errorMessages += data.toString();
    });

    pythonProcess.on('close', (code) => {
        if (code === 0 && !errorOccurred) {
            res.send(prediction);
        } else {
            console.error(`Python script exited with code ${code}: ${errorMessages}`);
            res.status(500).send("An error occurred during prediction.");
        }

        // Optionally delete the uploaded image after processing
        fs.unlink(imagePath, (err) => {
            if (err) console.error(`Failed to delete ${imagePath}: ${err}`);
        });
    });
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
