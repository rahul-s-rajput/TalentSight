const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const { app } = require('electron');

class Database {
  constructor() {
    this.db = new sqlite3.Database(
      path.join(app.getPath('userData'), 'talentsight.db'),
      (err) => {
        if (err) {
          console.error('Database connection error:', err);
        } else {
          console.log('Connected to the database');
          this.init();
        }
      }
    );
    
    // Enable foreign keys
    this.db.run('PRAGMA foreign_keys = ON');
  }

  init() {
    // Create tables if they don't exist
    this.db.serialize(() => {
      // Job presets table
      this.db.run(`
        CREATE TABLE IF NOT EXISTS job_presets (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
      `);

      // Skills for job presets
      this.db.run(`
        CREATE TABLE IF NOT EXISTS preset_skills (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          preset_id INTEGER NOT NULL,
          skill TEXT NOT NULL,
          FOREIGN KEY (preset_id) REFERENCES job_presets (id) ON DELETE CASCADE
        )
      `);

      // Custom criteria for job presets
      this.db.run(`
        CREATE TABLE IF NOT EXISTS preset_criteria (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          preset_id INTEGER NOT NULL,
          criteria TEXT NOT NULL,
          FOREIGN KEY (preset_id) REFERENCES job_presets (id) ON DELETE CASCADE
        )
      `);
    });
  }

  // Save a new job preset
  saveJobPreset(preset, callback) {
    const self = this; // Store reference to this
    
    // Use a transaction to ensure data integrity
    this.db.serialize(() => {
      this.db.run('BEGIN TRANSACTION');
      
      this.db.run(
        'INSERT INTO job_presets (name) VALUES (?)',
        [preset.name],
        function(err) {
          if (err) {
            console.error('Error saving job preset:', err);
            self.db.run('ROLLBACK');
            return callback(err);
          }
          
          const presetId = this.lastID; // This is the correct context for lastID
          console.log("Created preset with ID:", presetId);
          
          // Insert skills one by one to avoid prepare statement issues
          const insertSkill = (index) => {
            if (index >= preset.skills.length) {
              // All skills inserted, now insert criteria
              insertCriteria(0);
              return;
            }
            
            self.db.run(
              'INSERT INTO preset_skills (preset_id, skill) VALUES (?, ?)',
              [presetId, preset.skills[index]],
              (err) => {
                if (err) {
                  console.error('Error inserting skill:', err);
                  self.db.run('ROLLBACK');
                  return callback(err);
                }
                insertSkill(index + 1);
              }
            );
          };
          
          // Insert custom criteria one by one
          const insertCriteria = (index) => {
            if (!preset.customCriteria || index >= preset.customCriteria.length) {
              // All criteria inserted, commit transaction
              self.db.run('COMMIT', (err) => {
                if (err) {
                  console.error('Error committing transaction:', err);
                  self.db.run('ROLLBACK');
                  return callback(err);
                }
                callback(null, { id: presetId, name: preset.name });
              });
              return;
            }
            
            self.db.run(
              'INSERT INTO preset_criteria (preset_id, criteria) VALUES (?, ?)',
              [presetId, preset.customCriteria[index]],
              (err) => {
                if (err) {
                  console.error('Error inserting criteria:', err);
                  self.db.run('ROLLBACK');
                  return callback(err);
                }
                insertCriteria(index + 1);
              }
            );
          };
          
          // Start inserting skills
          insertSkill(0);
        }
      );
    });
  }

  // Get all job presets
  getAllJobPresets(callback) {
    this.db.all('SELECT * FROM job_presets ORDER BY name', [], (err, presets) => {
      if (err) {
        console.error('Error getting job presets:', err);
        return callback(err);
      }
      callback(null, presets);
    });
  }

  // Get a job preset with its skills and criteria
  getJobPresetDetails(presetId, callback) {
    this.db.get('SELECT * FROM job_presets WHERE id = ?', [presetId], (err, preset) => {
      if (err || !preset) {
        return callback(err || new Error('Preset not found'));
      }
      
      // Get skills
      this.db.all('SELECT skill FROM preset_skills WHERE preset_id = ?', [presetId], (err, skills) => {
        if (err) return callback(err);
        
        preset.skills = skills.map(s => s.skill);
        
        // Get criteria
        this.db.all('SELECT criteria FROM preset_criteria WHERE preset_id = ?', [presetId], (err, criteria) => {
          if (err) return callback(err);
          
          preset.customCriteria = criteria.map(c => c.criteria);
          callback(null, preset);
        });
      });
    });
  }

  // Delete a job preset
  deleteJobPreset(presetId, callback) {
    this.db.run(
      'DELETE FROM job_presets WHERE id = ?',
      [presetId],
      function(err) {
        if (err) {
          console.error('Error deleting job preset:', err);
          return callback(err);
        }
        callback(null, { success: true, rowsAffected: this.changes });
      }
    );
  }

  // Close the database connection
  close() {
    this.db.close((err) => {
      if (err) {
        console.error('Error closing database:', err);
      } else {
        console.log('Database connection closed');
      }
    });
  }
}

module.exports = Database; 