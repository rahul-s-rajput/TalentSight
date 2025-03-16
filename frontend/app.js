const { ipcRenderer } = require('electron');

document.addEventListener("DOMContentLoaded", () => {
  // Load the candidate analysis page by default
  loadPage('candidate-analysis');
  
  // Set up navigation
  setupNavigation();
});

function setupNavigation() {
  // Handle navigation menu clicks
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', function(e) {
      e.preventDefault();
      
      // Remove active class from all nav items
      document.querySelectorAll('.nav-item').forEach(navItem => {
        navItem.classList.remove('active');
      });
      
      // Add active class to clicked item
      this.classList.add('active');
      
      // Load the corresponding page
      const page = this.getAttribute('data-page');
      loadPage(page);
    });
  });
}

function loadPage(page) {
  const mainContent = document.getElementById('main-content');
  const loadingOverlay = document.getElementById('loading-overlay');
  
  // Show loading overlay
  loadingOverlay.classList.add('active');
  
  // Load the page content
  fetch(`pages/${page}.html`)
    .then(response => response.text())
    .then(html => {
      mainContent.innerHTML = html;
      
      // Initialize page-specific functionality
      if (page === 'candidate-analysis') {
        initCandidateAnalysis();
      } else if (page === 'create-preset') {
        initCreatePreset();
      } else if (page === 'job-presets') {
        initJobPresets();
      }
      
      // Hide loading overlay
      loadingOverlay.classList.remove('active');
    })
    .catch(error => {
      console.error('Error loading page:', error);
      mainContent.innerHTML = '<div class="error-message">Error loading page content</div>';
      loadingOverlay.classList.remove('active');
    });
}

function initCandidateAnalysis() {
  // DOM Elements
  const createPresetBtn = document.getElementById("create-preset-btn");
  const startAnalysisBtn = document.getElementById("start-analysis-btn");
  const videoUpload = document.getElementById("video-upload");
  const resumeUpload = document.getElementById("resume-upload");
  const videoInput = document.getElementById("video-file");
  const resumeInput = document.getElementById("resume-file");
  const videoStatus = document.getElementById("video-status");
  const resumeStatus = document.getElementById("resume-status");
  const jobPresetSelect = document.getElementById("job-preset");
  const loadingOverlay = document.getElementById("loading-overlay");

  // State

  let videoFile = null;
  let resumeFile = null;
  let selectedPresetId = "";


  // Navigation to create preset page
  createPresetBtn.addEventListener("click", () => {
    loadPage('create-preset');
  });


  // Replace or add the video selection functionality
  const originalVideoButton = videoUpload.querySelector(".upload-button");
  originalVideoButton.textContent = "Select Video File";
  
  // Replace standard file input with direct dialog call
  originalVideoButton.addEventListener("click", async (e) => {
    e.preventDefault();
    
    // Show loading state
    videoStatus.textContent = "Selecting file...";
    videoStatus.className = "upload-status info";
    
    try {
      // Use the native dialog
      const fileInfo = await ipcRenderer.invoke('select-video-file');
      
      if (fileInfo) {
        videoFile = fileInfo;
        // Ensure path property exists and log it for debugging
        console.log("Selected video file:", fileInfo);
        console.log("Video file path:", fileInfo.path);
        
        // Update UI to show selected file
        videoStatus.innerHTML = `
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg> ${fileInfo.name}`;
        videoStatus.className = "upload-status success";
        videoUpload.classList.add("uploaded");
        
        updateStartButton();
      } else {
        // User canceled
        videoStatus.textContent = "No file selected";
        videoStatus.className = "upload-status";
      }
    } catch (error) {
      console.error("Error selecting video:", error);
      videoStatus.textContent = "Error selecting file";
      videoStatus.className = "upload-status error";
    }
  });
  
  // Hide the original file input
  videoInput.style.display = "none";

  resumeInput.addEventListener("change", function() {
    if (this.files.length) {
      resumeFile = this.files[0];
      handleFileUpload(resumeFile, "resume", resumeStatus, resumeUpload);
    }
  });

  // Job preset selection
  jobPresetSelect.addEventListener("change", function() {
    selectedPresetId = this.value;

    updateStartButton();
  });

  // Update start analysis button state
  function updateStartButton() {

    if (videoFile && resumeFile && selectedPresetId) {

      startAnalysisBtn.disabled = false;
    } else {
      startAnalysisBtn.disabled = true;
    }
  }

  // Start analysis button
  startAnalysisBtn.addEventListener("click", () => {

    if (!videoFile || !selectedPresetId) {
      alert("Please upload a video and select a job preset");
      return;
    }

    // Validate video file path
    if (!videoFile.path) {
      console.error("Video file path is undefined:", videoFile);
      alert("Invalid video file path. Please select the video again.");
      return;
    }

    console.log("Starting analysis with video path:", videoFile.path);

    // Show loading overlay
    loadingOverlay.classList.add("active");
    loadingOverlay.querySelector("p").textContent = "Analyzing interview video...";
    
    // Get job preset details for the selected preset
    ipcRenderer.invoke('get-job-preset-details', selectedPresetId)
      .then(preset => {
        // Create temp files and paths
        return ipcRenderer.invoke('prepare-analysis', {
          videoPath: videoFile.path,
          resumePath: resumeFile ? resumeFile.path : null,
          preset: preset
        });
      })
      .then(config => {
        // Log the config being sent for analysis
        console.log("Analysis config:", config);
        // Start the video analysis
        return ipcRenderer.invoke('run-video-analysis', config);
      })
      .then(results => {
        // Hide loading overlay
        loadingOverlay.classList.remove("active");
        
        // Store results in session storage for results page
        sessionStorage.setItem('analysisResults', JSON.stringify(results));
        
        // Navigate to results page
        loadPage('analysis-results');
      })
      .catch(error => {
        console.error('Error during analysis:', error);
        loadingOverlay.classList.remove("active");
        alert("Error analyzing video: " + error.message);
      });

  });

  // Load job presets from database
  loadJobPresets();

  // Function to load job presets from database
  function loadJobPresets() {
    ipcRenderer.invoke('get-all-job-presets')
      .then(presets => {
        // Add presets to select dropdown
        presets.forEach(preset => {
          const option = document.createElement("option");
          option.value = preset.id;
          option.textContent = preset.name;
          jobPresetSelect.appendChild(option);
        });
      })
      .catch(error => {
        console.error('Error loading job presets:', error);
      });
  }

  
  // File upload handling
  function handleFileUpload(file, type, statusElement, uploadArea) {
    // Check file type
    let isValidFile = false;

    if (type === "video" && (file.type === "video/mp4" || file.type === "video/quicktime")) {
      isValidFile = true;
    } else if (
      type === "resume" &&
      (file.type === "application/pdf" ||
        file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    ) {
      isValidFile = true;
    }

    if (!isValidFile) {
      statusElement.textContent = "Invalid file format";
      statusElement.className = "upload-status error";
      return;
    }

    // Update UI
    statusElement.textContent = "File selected";
    statusElement.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg> ' +
      file.name;
    statusElement.className = "upload-status success";
    uploadArea.classList.add("uploaded");

    updateStartButton();
  }

  // File upload handling function
  function setupFileUpload(uploadArea, fileInput, statusElement, fileType) {
    // Drag and drop functionality
    uploadArea.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadArea.classList.add("drag-over");
    });

    uploadArea.addEventListener("dragleave", () => {
      uploadArea.classList.remove("drag-over");
    });

    uploadArea.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadArea.classList.remove("drag-over");

      const files = e.dataTransfer.files;
      if (files.length) {
        fileInput.files = files; // Update the file input
        const event = new Event('change'); // Create a change event
        fileInput.dispatchEvent(event); // Dispatch it to trigger the change handler
      }
    });

    // Click to upload functionality
    uploadArea.querySelector(".upload-button").addEventListener("click", () => {
      fileInput.click();
    });
  }

  // IMPORTANT: Actually call the setup function for drag & drop handling
  setupFileUpload(videoUpload, videoInput, videoStatus, "video");
  setupFileUpload(resumeUpload, resumeInput, resumeStatus, "resume");

}

function initCreatePreset() {
  // DOM Elements
  const cancelPresetBtn = document.getElementById("cancel-preset-btn");
  const savePresetBtn = document.getElementById("save-preset-btn");
  const addSkillBtn = document.getElementById("add-skill-btn");
  const skillsInput = document.getElementById("skills-input");
  const skillsContainer = document.getElementById("skills-container");
  const addCriteriaBtn = document.getElementById("add-criteria-btn");
  const customCriteriaInput = document.getElementById("custom-criteria");
  const customCriteriaContainer = document.getElementById("custom-criteria-container");
  const loadingOverlay = document.getElementById("loading-overlay");

  // State
  const skills = [];
  const customCriteria = [];

  // Skill suggestions
  const skillSuggestions = [
    "JavaScript", "React", "Node.js", "Python", "Java", "C#", "SQL",
    "Data Analysis", "Machine Learning", "Project Management", "UI/UX Design",
    "Communication", "Leadership", "Problem Solving", "Teamwork", "Agile",
    "DevOps", "Cloud Computing", "AWS", "Azure", "Docker", "Kubernetes",
  ];

  // Navigation back to candidate analysis
  cancelPresetBtn.addEventListener("click", () => {
    loadPage('candidate-analysis');
  });

  // Save preset
  savePresetBtn.addEventListener("click", () => {
    const presetName = document.getElementById("preset-name").value;
    if (!presetName) {
      alert("Please enter a preset name");
      return;
    }

    if (skills.length === 0) {
      alert("Please add at least one required skill");
      return;
    }

    // Show loading overlay
    loadingOverlay.classList.add("active");

    // Get selected evaluation criteria
    const evaluationCriteria = [];
    document.querySelectorAll('.criteria-item input[type="checkbox"]:checked').forEach(checkbox => {
      evaluationCriteria.push(checkbox.id);
    });

    // Add custom criteria
    evaluationCriteria.push(...customCriteria);

    // Create preset object
    const preset = {
      name: presetName,
      skills: skills,
      evaluationCriteria: evaluationCriteria
    };

    // Save preset to database
    ipcRenderer.invoke('save-job-preset', preset)
      .then(result => {
        // Hide loading overlay
        loadingOverlay.classList.remove("active");
        
        // Navigate back to candidate analysis
        loadPage('candidate-analysis');
      })
      .catch(error => {
        console.error('Error saving job preset:', error);
        loadingOverlay.classList.remove("active");
        alert("Error saving job preset. Please try again.");
      });
  });

  // Skills management
  addSkillBtn.addEventListener("click", () => {
    addSkill();
  });

  skillsInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addSkill();
    }
  });

  function addSkill() {
    const skill = skillsInput.value.trim();
    if (skill && !skills.includes(skill)) {
      skills.push(skill);
      renderSkills();
      skillsInput.value = "";
    }
  }

  function renderSkills() {
    skillsContainer.innerHTML = "";
    skills.forEach((skill) => {
      const skillTag = document.createElement("div");
      skillTag.className = "skill-tag";
      skillTag.innerHTML = `
        ${skill}
        <span class="remove-tag" data-skill="${skill}">×</span>
      `;
      skillsContainer.appendChild(skillTag);
    });

    // Add event listeners to remove buttons
    document.querySelectorAll(".skill-tag .remove-tag").forEach((btn) => {
      btn.addEventListener("click", function () {
        const skillToRemove = this.getAttribute("data-skill");
        const index = skills.indexOf(skillToRemove);
        if (index !== -1) {
          skills.splice(index, 1);
          renderSkills();
        }
      });
    });
  }

  // Custom criteria management
  addCriteriaBtn.addEventListener("click", () => {
    addCustomCriteria();
  });

  customCriteriaInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addCustomCriteria();
    }
  });

  function addCustomCriteria() {
    const criteria = customCriteriaInput.value.trim();
    if (criteria && !customCriteria.includes(criteria)) {
      customCriteria.push(criteria);
      renderCustomCriteria();
      customCriteriaInput.value = "";
    }
  }

  function renderCustomCriteria() {
    customCriteriaContainer.innerHTML = "";
    customCriteria.forEach((criteria) => {
      const criteriaTag = document.createElement("div");
      criteriaTag.className = "criteria-tag";
      criteriaTag.innerHTML = `
        ${criteria}
        <span class="remove-tag" data-criteria="${criteria}">×</span>
      `;
      customCriteriaContainer.appendChild(criteriaTag);
    });

    // Add event listeners to remove buttons
    document.querySelectorAll(".criteria-tag .remove-tag").forEach((btn) => {
      btn.addEventListener("click", function () {
        const criteriaToRemove = this.getAttribute("data-criteria");
        const index = customCriteria.indexOf(criteriaToRemove);
        if (index !== -1) {
          customCriteria.splice(index, 1);
          renderCustomCriteria();
        }
      });
    });
  }

  // Initialize skill suggestions
  const skillsSuggestions = document.getElementById("skills-suggestions");

  skillsInput.addEventListener("input", function () {
    const value = this.value.trim().toLowerCase();
    if (value.length < 2) {
      skillsSuggestions.innerHTML = "";
      return;
    }

    const filteredSuggestions = skillSuggestions
      .filter((skill) => skill.toLowerCase().includes(value) && !skills.includes(skill))
      .slice(0, 5);

    if (filteredSuggestions.length) {
      skillsSuggestions.innerHTML = "";
      filteredSuggestions.forEach((suggestion) => {
        const suggestionItem = document.createElement("div");
        suggestionItem.className = "suggestion-item";
        suggestionItem.textContent = suggestion;
        suggestionItem.addEventListener("click", () => {
          skillsInput.value = "";
          if (!skills.includes(suggestion)) {
            skills.push(suggestion);
            renderSkills();
          }
          skillsSuggestions.innerHTML = "";
        });
        skillsSuggestions.appendChild(suggestionItem);
      });
    } else {
      skillsSuggestions.innerHTML = "";
    }
  });

  // Clear suggestions when clicking outside
  document.addEventListener("click", (e) => {
    if (!skillsInput.contains(e.target) && !skillsSuggestions.contains(e.target)) {
      skillsSuggestions.innerHTML = "";
    }
  });
}

function initJobPresets() {
  // DOM Elements
  const presetsList = document.getElementById('presets-list');
  const presetDetailsContainer = document.getElementById('preset-details-container');
  const presetDetails = document.getElementById('preset-details').querySelector('.panel-content');
  const createNewPresetBtn = document.getElementById('create-new-preset-btn');
  const closeDetailsBtn = document.getElementById('close-details-btn');
  const deleteModal = document.getElementById('delete-modal');
  const cancelDeleteBtn = document.getElementById('cancel-delete-btn');
  const confirmDeleteBtn = document.getElementById('confirm-delete-btn');
  
  let selectedPresetId = null;
  
  // Load all job presets
  loadAllPresets();
  
  // Event Listeners
  createNewPresetBtn.addEventListener('click', () => {
    loadPage('create-preset');
  });
  
  closeDetailsBtn.addEventListener('click', () => {
    presetDetailsContainer.style.display = 'none';
  });
  
  cancelDeleteBtn.addEventListener('click', () => {
    deleteModal.style.display = 'none';
  });
  
  confirmDeleteBtn.addEventListener('click', () => {
    if (selectedPresetId) {
      deletePreset(selectedPresetId);
    }
  });
  
  // Close modal when clicking outside
  window.addEventListener('click', (event) => {
    if (event.target === deleteModal) {
      deleteModal.style.display = 'none';
    }
  });
  
  function loadAllPresets() {
    // Show loading indicator
    presetsList.innerHTML = `
      <tr>
        <td colspan="3" class="loading-cell">
          <div class="loading-spinner">Loading presets...</div>
        </td>
      </tr>
    `;
    
    ipcRenderer.invoke('get-all-job-presets')
      .then(presets => {
        if (presets.length === 0) {
          presetsList.innerHTML = `
            <tr>
              <td colspan="3" class="empty-state">
                No presets found. Create your first job preset!
              </td>
            </tr>
          `;
          return;
        }
        
        // Render presets list
        presetsList.innerHTML = '';
        presets.forEach(preset => {
          const row = document.createElement('tr');
          row.className = 'preset-row';
          row.setAttribute('data-id', preset.id);
          row.innerHTML = `
            <td class="preset-name">${preset.name}</td>
            <td class="preset-date">Created: ${formatDate(preset.created_at)}</td>
            <td class="preset-actions">
              <button class="icon-button view-preset" title="View Details">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
              </button>
              <button class="icon-button edit-preset" title="Edit Preset">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
              </button>
              <button class="icon-button delete-preset" title="Delete Preset">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
              </button>
            </td>
          `;
          presetsList.appendChild(row);
          
          // Add event listeners to buttons
          row.querySelector('.view-preset').addEventListener('click', () => {
            loadPresetDetails(preset.id);
          });
          
          // Make the entire row clickable to view details
          row.querySelector('.preset-name').addEventListener('click', () => {
            loadPresetDetails(preset.id);
          });
          
          row.querySelector('.edit-preset').addEventListener('click', () => {
            // Navigate to edit page with preset ID
            // This will need to be implemented in the future
            alert('Edit functionality will be implemented in future updates');
          });
          
          row.querySelector('.delete-preset').addEventListener('click', () => {
            selectedPresetId = preset.id;
            deleteModal.style.display = 'block';
          });
        });
      })
      .catch(error => {
        console.error('Error loading job presets:', error);
        presetsList.innerHTML = `
          <tr>
            <td colspan="3" class="error-message">
              Error loading presets. Please try again.
            </td>
          </tr>
        `;
      });
  }
  
  function loadPresetDetails(presetId) {
    // Show loading indicator
    presetDetails.innerHTML = '<div class="loading-spinner">Loading details...</div>';
    presetDetailsContainer.style.display = 'block';
    
    // Highlight selected preset
    document.querySelectorAll('.preset-row').forEach(row => {
      row.classList.remove('selected');
      if (row.getAttribute('data-id') == presetId) {
        row.classList.add('selected');
      }
    });
    
    ipcRenderer.invoke('get-job-preset-details', presetId)
      .then(preset => {
        presetDetails.innerHTML = `
          <div class="details-header">
            <h3>${preset.name}</h3>
            <p class="details-meta">Created: ${formatDate(preset.created_at)}</p>
          </div>
          
          <div class="details-section">
            <h4>Required Skills</h4>
            <div class="skills-list">
              ${preset.skills.map(skill => `<div class="skill-tag">${skill}</div>`).join('') || '<p>No skills specified</p>'}
            </div>
          </div>
          
          <div class="details-section">
            <h4>Custom Evaluation Criteria</h4>
            <div class="criteria-list">
              ${preset.customCriteria.map(criteria => `<div class="criteria-item">${criteria}</div>`).join('') || '<p>No custom criteria specified</p>'}
            </div>
          </div>
        `;
      })
      .catch(error => {
        console.error('Error loading preset details:', error);
        presetDetails.innerHTML = '<div class="error-message">Error loading preset details</div>';
      });
  }
  
  function deletePreset(presetId) {
    ipcRenderer.invoke('delete-job-preset', presetId)
      .then(() => {
        // Close modal and reload presets
        deleteModal.style.display = 'none';
        presetDetailsContainer.style.display = 'none';
        loadAllPresets();
      })
      .catch(error => {
        console.error('Error deleting preset:', error);
        alert('Error deleting preset. Please try again.');
        deleteModal.style.display = 'none';
      });
  }
  
  function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    });
  }
}


document.addEventListener('DOMContentLoaded', function() {
  // The dynamic data you provided
  const data = {
      "STAR_Method_Scores": {
          "Situation": "7.9/10",
          "Task": "9.5/10",
          "Action": "10/10",
          "Result": "9.4/10",
          "Average": "9.2/10"
      },
      "Three_Cs_Scores": {
          "Credibility": "3.8/10",
          "Competence": "7.1/10",
          "Confidence": "7.7/10",
          "Average": "6.2/10"
      },
      "Overall_Score": "7.7/10",
      "Feedback": {
          "Overall_Evaluation_Score": "7.7/10",
          "STAR_Method_Analysis": {
              "Strengths": [
                  "The candidate demonstrated good Situation, Task, Action, Result.",
                  "They provided clear context and background for their examples.",
                  "They effectively communicated goals and responsibilities.",
                  "They detailed specific actions and strategies they implemented.",
                  "They effectively quantified outcomes and highlighted achievements."
              ]
          },
          "Three_Cs_Analysis": {
              "Strengths": [
                  "The candidate showed strong Competence, Confidence.",
                  "They demonstrated strong skills and technical knowledge.",
                  "They communicated with conviction and certainty."
              ],
              "Areas_for_Improvement": [
                  "The candidate could strengthen their Credibility.",
                  "They should provide more specific evidence of their experience and achievements."
              ]
          },
          "Summary": [
              "This candidate effectively communicates their experiences and skills, using structured responses with clear examples.",
              "They demonstrate significant alignment with the job requirements and show strong potential in the key areas evaluated."
          ]
      }
  };

  // Set overall score
  // document.getElementById('overall-score').textContent = data.Overall_Score.split('/')[0];

  // Function to update score bar and text
  function updateScore(scoreType, category, scoreData) {
      const scoreValue = parseFloat(scoreData.split('/')[0]);
      const scoreId = `${category.toLowerCase()}-score`;
      const barId = `${category.toLowerCase()}-bar`;
      
      // Update score text
      // document.getElementById(scoreId).textContent = scoreData;

                const element = document.getElementById(scoreId);
if (element) {
    element.textContent = scoreData;
} else {
    console.error(`Element with ID '${scoreId}' not found`);
}
      
      // Update score bar
      const barElement = document.getElementById(barId);
      barElement.style.width = (scoreValue * 10) + '%';
      
      // Set color based on score
      if (scoreValue >= 7) {
          barElement.style.backgroundColor = '#4caf50'; // Green
      } else if (scoreValue >= 5) {
          barElement.style.backgroundColor = '#ff9800'; // Orange
      } else {
          barElement.style.backgroundColor = '#f44336'; // Red
      }
  }

  // Update STAR Method scores
  updateScore('STAR', 'situation', data.STAR_Method_Scores.Situation);
  updateScore('STAR', 'task', data.STAR_Method_Scores.Task);
  updateScore('STAR', 'action', data.STAR_Method_Scores.Action);
  updateScore('STAR', 'result', data.STAR_Method_Scores.Result);
  document.getElementById('star-average').textContent = data.STAR_Method_Scores.Average;

  // Update Three Cs scores
  updateScore('ThreeCs', 'credibility', data.Three_Cs_Scores.Credibility);
  updateScore('ThreeCs', 'competence', data.Three_Cs_Scores.Competence);
  updateScore('ThreeCs', 'confidence', data.Three_Cs_Scores.Confidence);
  document.getElementById('three-cs-average').textContent = data.Three_Cs_Scores.Average;

  // Set feedback content
  function populateList(elementId, items) {
    console.log('est')
      if(elementId === "star-strengths") {
        console.log(elementId, items)
      }
      const list = document.getElementById(elementId);
      list.innerHTML = '';
      items.forEach(item => {
          const li = document.createElement('li');
          li.textContent = item;
          list.appendChild(li);
      });
  }

  populateList('star-strengths', data.Feedback.STAR_Method_Analysis.Strengths);
  populateList('three-cs-strengths', data.Feedback.Three_Cs_Analysis.Strengths);
  populateList('three-cs-improvements', data.Feedback.Three_Cs_Analysis.Areas_for_Improvement);

  // Set summary
  const summaryEl = document.getElementById('summary');
  summaryEl.innerHTML = '';
  data.Feedback.Summary.forEach(paragraph => {
      const p = document.createElement('p');
      p.textContent = paragraph;
      summaryEl.appendChild(p);
  });
});