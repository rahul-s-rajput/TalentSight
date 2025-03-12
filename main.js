const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const Database = require('./database');

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
}

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
