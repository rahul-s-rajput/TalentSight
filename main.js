
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const Database = require('./database');
const fs = require('fs');
const { spawn } = require('child_process');
const tmp = require('tmp');


let db;

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    }
  });

  win.loadFile(path.join(__dirname, 'frontend', 'index.html'));
}

app.whenReady().then(() => {
  // Initialize database
  db = new Database();
  
  // Set up IPC handlers for database operations
  setupIpcHandlers();
  
  createWindow();
});

function setupIpcHandlers() {
  // Save a new job preset
  ipcMain.handle('save-job-preset', async (event, preset) => {
    return new Promise((resolve, reject) => {
      db.saveJobPreset(preset, (err, result) => {
        if (err) reject(err);
        else resolve(result);
      });
    });
  });

  // Get all job presets
  ipcMain.handle('get-all-job-presets', async () => {
    return new Promise((resolve, reject) => {
      db.getAllJobPresets((err, presets) => {
        if (err) reject(err);
        else resolve(presets);
      });
    });
  });

  // Get job preset details
  ipcMain.handle('get-job-preset-details', async (event, presetId) => {
    return new Promise((resolve, reject) => {
      db.getJobPresetDetails(presetId, (err, preset) => {
        if (err) reject(err);
        else resolve(preset);
      });
    });
  });

  // Delete job preset
  ipcMain.handle('delete-job-preset', async (event, presetId) => {
    return new Promise((resolve, reject) => {
      db.deleteJobPreset(presetId, (err, result) => {
        if (err) reject(err);
        else resolve(result);
      });
    });
  });


  // Select video file
  ipcMain.handle('select-video-file', async (event) => {
    const { canceled, filePaths } = await dialog.showOpenDialog({
      properties: ['openFile'],
      filters: [
        { name: 'Videos', extensions: ['mp4', 'mov', 'avi', 'mkv'] }
      ]
    });
    
    if (canceled || filePaths.length === 0) {
      return null;
    }
    
    const filePath = filePaths[0];
    const fileName = path.basename(filePath);
    
    return {
      path: filePath,
      name: fileName
    };
  });
}

// IPC handlers for video analysis
ipcMain.handle('prepare-analysis', async (event, data) => {
  // Create a temporary directory for analysis outputs
  const tmpDir = tmp.dirSync({ prefix: 'talentsight-analysis-' });
  
  // Create paths for output files
  const outputVideoPath = path.join(tmpDir.name, 'processed_video.mp4');
  const outputTranscriptPath = path.join(tmpDir.name, 'transcript.txt');
  const outputGazeReportPath = path.join(tmpDir.name, 'gaze_report.txt');
  const outputEvaluationPath = path.join(tmpDir.name, 'evaluation.json');
  
  // Create a job description file if we have a job preset
  let jobDescriptionPath = null;
  if (data.preset) {
    jobDescriptionPath = path.join(tmpDir.name, 'job_description.txt');
    
    // Create a job description from the preset
    const jobDescription = `
      Job Title: ${data.preset.name}
      
      Required Skills:
      ${data.preset.skills.join(', ')}
      
      Evaluation Criteria:
      ${data.preset.customCriteria ? data.preset.customCriteria.join(', ') : 'Standard evaluation criteria'}
    `;
    
    fs.writeFileSync(jobDescriptionPath, jobDescription);
  }
  
  return {
    videoPath: data.videoPath,
    resumePath: data.resumePath,
    jobDescriptionPath,
    outputVideoPath,
    outputTranscriptPath,
    outputGazeReportPath,
    outputEvaluationPath,
    tempDir: tmpDir.name
  };
});

ipcMain.handle('run-video-analysis', async (event, config) => {
  return new Promise((resolve, reject) => {
    // More extensive validation of videoPath
    if (!config.videoPath || config.videoPath === "undefined") {
      return reject(new Error('No valid video path provided for analysis. Please try uploading the file again.'));
    }
    
    // Check if the file exists
    try {
      if (!fs.existsSync(config.videoPath)) {
        return reject(new Error(`Video file not found at path: ${config.videoPath}`));
      }
    } catch (err) {
      return reject(new Error(`Error accessing video file: ${err.message}`));
    }
    
    // Build command to run video_analysis.py
    const scriptPath = path.join(__dirname, 'backend', 'video_analysis.py');
    
    const args = [
      scriptPath,
      '--video', config.videoPath,
      '--output_video', config.outputVideoPath,
      '--output_transcript', config.outputTranscriptPath,
      '--output_gaze_report', config.outputGazeReportPath,
      '--output_evaluation', config.outputEvaluationPath
    ];
    
    // Add job description if available
    if (config.jobDescriptionPath) {
      args.push('--job_description', config.jobDescriptionPath);
    }
    
    console.log('Running video analysis with command:', `python ${args.join(' ')}`);
    
    // Spawn python process
    const pythonProcess = spawn('python', args, { 
      stdio: ['ignore', 'pipe', 'pipe']
    });
    
    let stdoutData = '';
    let stderrData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      stdoutData += data.toString();
      console.log(`Analysis stdout: ${data}`);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      stderrData += data.toString();
      console.error(`Analysis stderr: ${data}`);
    });
    
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`Analysis process exited with code ${code}: ${stderrData}`));
      }
      
      try {
        // Read evaluation results
        const evaluation_results = JSON.parse(fs.readFileSync(config.outputEvaluationPath, 'utf8'));
        
        // Read transcript
        const transcript = fs.readFileSync(config.outputTranscriptPath, 'utf8');
        
        // Read gaze report
        const gazeReport = fs.readFileSync(config.outputGazeReportPath, 'utf8');
        
        // Parse segments from transcript
        const segments = parseTranscript(transcript);
        
        // Parse gaze report
        const gaze_report = parseGazeReport(gazeReport);
        
        resolve({
          segments,
          evaluation_results,
          gaze_report,
          processedVideoPath: config.outputVideoPath
        });
      } catch (err) {
        reject(new Error(`Error parsing analysis results: ${err.message}`));
      }
    });
  });
});

// Helper function to parse transcript into segments
function parseTranscript(transcript) {
  const segments = [];
  const lines = transcript.split('\n');
  
  for (const line of lines) {
    const match = line.match(/\[(\d+\.\d+)s - (\d+\.\d+)s\] ([^:]+): (.*)/);
    if (match) {
      segments.push({
        start: parseFloat(match[1]),
        end: parseFloat(match[2]),
        speaker: match[3],
        text: match[4]
      });
    }
  }
  
  return segments;
}

// Helper function to parse gaze report
function parseGazeReport(reportText) {
  const report = {};
  
  // Extract suspicion score
  const suspicionMatch = reportText.match(/Suspicion Score: (\d+)\/100/);
  if (suspicionMatch) {
    report.suspicion_score = parseInt(suspicionMatch[1]);
  }
  
  // Extract suspicion level
  const levelMatch = reportText.match(/Suspicion Level: ([A-Za-z]+)/);
  if (levelMatch) {
    report.suspicion_level = levelMatch[1];
  }
  
  // Extract behavior assessment
  const assessmentMatch = reportText.match(/Behavior Assessment: ([^\n]+)/);
  if (assessmentMatch) {
    report.behavior_assessment = assessmentMatch[1];
  }
  
  // Extract reading percentage
  const readingMatch = reportText.match(/Reading Duration: \d+\.\d+ seconds \((\d+\.\d+)%\)/);
  if (readingMatch) {
    report.reading_percentage = parseFloat(readingMatch[1]);
  }
  
  return report;
}

// Export report as PDF
ipcMain.handle('export-analysis-report', async (event, results) => {
  const { filePath } = await dialog.showSaveDialog({
    title: 'Save Analysis Report',
    defaultPath: path.join(app.getPath('documents'), 'analysis-report.pdf'),
    filters: [{ name: 'PDF Files', extensions: ['pdf'] }]
  });
  
  if (!filePath) {
    throw new Error('Export cancelled');
  }
  
  // Call a Python script to generate the PDF report
  const scriptPath = path.join(__dirname, 'backend', 'generate_report.py');
  
  // Create a temporary JSON file with the results
  const tempResultsPath = tmp.tmpNameSync({ postfix: '.json' });
  fs.writeFileSync(tempResultsPath, JSON.stringify(results));
  
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      scriptPath,
      '--input', tempResultsPath,
      '--output', filePath
    ]);
    
    pythonProcess.on('close', (code) => {
      // Clean up the temporary file
      fs.unlinkSync(tempResultsPath);
      
      if (code !== 0) {
        reject(new Error(`Report generation failed with code ${code}`));
      } else {
        resolve(filePath);
      }
    });
  });
});


app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('will-quit', () => {
  // Close database connection when app is about to quit
  if (db) {
    db.close();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
