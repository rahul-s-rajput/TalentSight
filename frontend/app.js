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
  let videoUploaded = false;
  let resumeUploaded = false;
  let presetSelected = false;

  // Navigation to create preset page
  createPresetBtn.addEventListener("click", () => {
    loadPage('create-preset');
  });

  // File upload handling
  setupFileUpload(videoUpload, videoInput, videoStatus, "video");
  setupFileUpload(resumeUpload, resumeInput, resumeStatus, "resume");

  // Job preset selection
  jobPresetSelect.addEventListener("change", function () {
    presetSelected = this.value !== "";
    updateStartButton();
  });

  // Update start analysis button state
  function updateStartButton() {
    if (videoUploaded && resumeUploaded && presetSelected) {
      startAnalysisBtn.disabled = false;
    } else {
      startAnalysisBtn.disabled = true;
    }
  }

  // Start analysis button
  startAnalysisBtn.addEventListener("click", () => {
    loadingOverlay.classList.add("active");

    // Simulate analysis process
    setTimeout(() => {
      loadingOverlay.classList.remove("active");
      alert("Analysis completed successfully! Redirecting to results page...");
    }, 3000);
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
      handleFileUpload(files[0], fileType);
    }
  });

  // Click to upload functionality
  uploadArea.querySelector(".upload-button").addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", function () {
    if (this.files.length) {
      handleFileUpload(this.files[0], fileType);
    }
  });

  function handleFileUpload(file, type) {
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

    // Simulate file upload
    statusElement.textContent = "Uploading...";
    statusElement.className = "upload-status";

    setTimeout(() => {
      statusElement.innerHTML =
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg> ' +
        file.name;
      statusElement.className = "upload-status success";
      uploadArea.classList.add("uploaded");

      if (type === "video") {
        window.videoUploaded = true;
      } else if (type === "resume") {
        window.resumeUploaded = true;
      }

      updateStartButton();
    }, 1500);
  }

  function updateStartButton() {
    const startAnalysisBtn = document.getElementById("start-analysis-btn");
    if (window.videoUploaded && window.resumeUploaded && window.presetSelected) {
      startAnalysisBtn.disabled = false;
    } else {
      startAnalysisBtn.disabled = true;
    }
  }
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

