const { ipcRenderer } = require('electron');

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const createPresetBtn = document.getElementById("create-preset-btn")
  const cancelPresetBtn = document.getElementById("cancel-preset-btn")
  const savePresetBtn = document.getElementById("save-preset-btn")
  const startAnalysisBtn = document.getElementById("start-analysis-btn")
  const uploadInterface = document.getElementById("upload-interface")
  const createPresetInterface = document.getElementById("create-preset-interface")
  const loadingOverlay = document.getElementById("loading-overlay")
  const videoUpload = document.getElementById("video-upload")
  const resumeUpload = document.getElementById("resume-upload")
  const videoInput = document.getElementById("video-file")
  const resumeInput = document.getElementById("resume-file")
  const videoStatus = document.getElementById("video-status")
  const resumeStatus = document.getElementById("resume-status")
  const jobPresetSelect = document.getElementById("job-preset")
  const addSkillBtn = document.getElementById("add-skill-btn")
  const skillsInput = document.getElementById("skills-input")
  const skillsContainer = document.getElementById("skills-container")
  const addCriteriaBtn = document.getElementById("add-criteria-btn")
  const customCriteriaInput = document.getElementById("custom-criteria")
  const customCriteriaContainer = document.getElementById("custom-criteria-container")

  // State
  let videoUploaded = false
  let resumeUploaded = false
  let presetSelected = false
  const skills = []
  const customCriteria = []

  // Skill suggestions
  const skillSuggestions = [
    "JavaScript",
    "React",
    "Node.js",
    "Python",
    "Java",
    "C#",
    "SQL",
    "Data Analysis",
    "Machine Learning",
    "Project Management",
    "UI/UX Design",
    "Communication",
    "Leadership",
    "Problem Solving",
    "Teamwork",
    "Agile",
    "DevOps",
    "Cloud Computing",
    "AWS",
    "Azure",
    "Docker",
    "Kubernetes",
  ]

  // Navigation between interfaces
  createPresetBtn.addEventListener("click", () => {
    uploadInterface.classList.remove("active")
    createPresetInterface.classList.add("active")
  })

  cancelPresetBtn.addEventListener("click", () => {
    createPresetInterface.classList.remove("active")
    uploadInterface.classList.add("active")
  })

  async function loadJobPresets() {
    try {
      const presets = await ipcRenderer.invoke('get-all-job-presets');
      
      // Clear existing options except the default one
      while (jobPresetSelect.options.length > 1) {
        jobPresetSelect.remove(1);
      }
      
      // Add presets from database
      presets.forEach(preset => {
        const option = document.createElement('option');
        option.value = preset.id;
        option.textContent = preset.name;
        jobPresetSelect.appendChild(option);
      });
    } catch (error) {
      console.error('Error loading job presets:', error);
    }
  }

  savePresetBtn.addEventListener("click", async () => {
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

    try {
      // Save preset to database
      const preset = {
        name: presetName,
        skills: skills,
        customCriteria: customCriteria
      };
      
      const savedPreset = await ipcRenderer.invoke('save-job-preset', preset);
      
      // Reload presets from database
      await loadJobPresets();
      
      // Select the new preset
      jobPresetSelect.value = savedPreset.id;
      presetSelected = true;
      updateStartButton();

      // Reset form
      document.getElementById("preset-name").value = "";
      skillsContainer.innerHTML = "";
      skills.length = 0;
      customCriteriaContainer.innerHTML = "";
      customCriteria.length = 0;

      // Switch back to upload interface
      createPresetInterface.classList.remove("active");
      uploadInterface.classList.add("active");
    } catch (error) {
      console.error('Error saving job preset:', error);
      alert('Failed to save job preset. Please try again.');
    } finally {
      // Hide loading overlay
      loadingOverlay.classList.remove("active");
    }
  });

  // File upload handling
  function setupFileUpload(uploadArea, fileInput, statusElement, fileType) {
    // Drag and drop functionality
    uploadArea.addEventListener("dragover", (e) => {
      e.preventDefault()
      uploadArea.classList.add("drag-over")
    })

    uploadArea.addEventListener("dragleave", () => {
      uploadArea.classList.remove("drag-over")
    })

    uploadArea.addEventListener("drop", (e) => {
      e.preventDefault()
      uploadArea.classList.remove("drag-over")

      const files = e.dataTransfer.files
      if (files.length) {
        handleFileUpload(files[0], fileType)
      }
    })

    // Click to upload functionality
    uploadArea.querySelector(".upload-button").addEventListener("click", () => {
      fileInput.click()
    })

    fileInput.addEventListener("change", function () {
      if (this.files.length) {
        handleFileUpload(this.files[0], fileType)
      }
    })

    function handleFileUpload(file, type) {
      // Check file type
      let isValidFile = false

      if (type === "video" && (file.type === "video/mp4" || file.type === "video/quicktime")) {
        isValidFile = true
      } else if (
        type === "resume" &&
        (file.type === "application/pdf" ||
          file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
      ) {
        isValidFile = true
      }

      if (!isValidFile) {
        statusElement.textContent = "Invalid file format"
        statusElement.className = "upload-status error"
        return
      }

      // Simulate file upload
      statusElement.textContent = "Uploading..."
      statusElement.className = "upload-status"

      setTimeout(() => {
        statusElement.innerHTML =
          '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg> ' +
          file.name
        statusElement.className = "upload-status success"
        uploadArea.classList.add("uploaded")

        if (type === "video") {
          videoUploaded = true
        } else if (type === "resume") {
          resumeUploaded = true
        }

        updateStartButton()
      }, 1500)
    }
  }

  // Setup file uploads
  setupFileUpload(videoUpload, videoInput, videoStatus, "video")
  setupFileUpload(resumeUpload, resumeInput, resumeStatus, "resume")

  // Job preset selection
  jobPresetSelect.addEventListener("change", function () {
    presetSelected = this.value !== ""
    updateStartButton()
  })

  // Update start analysis button state
  function updateStartButton() {
    if (videoUploaded && resumeUploaded && presetSelected) {
      startAnalysisBtn.disabled = false
    } else {
      startAnalysisBtn.disabled = true
    }
  }

  // Start analysis button
  startAnalysisBtn.addEventListener("click", () => {
    loadingOverlay.classList.add("active")

    // Simulate analysis process
    setTimeout(() => {
      loadingOverlay.classList.remove("active")
      alert("Analysis completed successfully! Redirecting to results page...")
    }, 3000)
  })

  // Skills management
  addSkillBtn.addEventListener("click", () => {
    addSkill()
  })

  skillsInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      e.preventDefault()
      addSkill()
    }
  })

  function addSkill() {
    const skill = skillsInput.value.trim()
    if (skill && !skills.includes(skill)) {
      skills.push(skill)
      renderSkills()
      skillsInput.value = ""
    }
  }

  function renderSkills() {
    skillsContainer.innerHTML = ""
    skills.forEach((skill) => {
      const skillTag = document.createElement("div")
      skillTag.className = "skill-tag"
      skillTag.innerHTML = `
        ${skill}
        <span class="remove-tag" data-skill="${skill}">×</span>
      `
      skillsContainer.appendChild(skillTag)
    })

    // Add event listeners to remove buttons
    document.querySelectorAll(".skill-tag .remove-tag").forEach((btn) => {
      btn.addEventListener("click", function () {
        const skillToRemove = this.getAttribute("data-skill")
        const index = skills.indexOf(skillToRemove)
        if (index !== -1) {
          skills.splice(index, 1)
          renderSkills()
        }
      })
    })
  }

  // Custom criteria management
  addCriteriaBtn.addEventListener("click", () => {
    addCustomCriteria()
  })

  customCriteriaInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      e.preventDefault()
      addCustomCriteria()
    }
  })

  function addCustomCriteria() {
    const criteria = customCriteriaInput.value.trim()
    if (criteria && !customCriteria.includes(criteria)) {
      customCriteria.push(criteria)
      renderCustomCriteria()
      customCriteriaInput.value = ""
    }
  }

  function renderCustomCriteria() {
    customCriteriaContainer.innerHTML = ""
    customCriteria.forEach((criteria) => {
      const criteriaTag = document.createElement("div")
      criteriaTag.className = "criteria-tag"
      criteriaTag.innerHTML = `
        ${criteria}
        <span class="remove-tag" data-criteria="${criteria}">×</span>
      `
      customCriteriaContainer.appendChild(criteriaTag)
    })

    // Add event listeners to remove buttons
    document.querySelectorAll(".criteria-tag .remove-tag").forEach((btn) => {
      btn.addEventListener("click", function () {
        const criteriaToRemove = this.getAttribute("data-criteria")
        const index = customCriteria.indexOf(criteriaToRemove)
        if (index !== -1) {
          customCriteria.splice(index, 1)
          renderCustomCriteria()
        }
      })
    })
  }

  // Initialize skill suggestions
  const skillsSuggestions = document.getElementById("skills-suggestions")

  skillsInput.addEventListener("input", function () {
    const value = this.value.trim().toLowerCase()
    if (value.length < 2) {
      skillsSuggestions.innerHTML = ""
      return
    }

    const filteredSuggestions = skillSuggestions
      .filter((skill) => skill.toLowerCase().includes(value) && !skills.includes(skill))
      .slice(0, 5)

    if (filteredSuggestions.length) {
      skillsSuggestions.innerHTML = ""
      filteredSuggestions.forEach((suggestion) => {
        const suggestionItem = document.createElement("div")
        suggestionItem.className = "suggestion-item"
        suggestionItem.textContent = suggestion
        suggestionItem.addEventListener("click", () => {
          skillsInput.value = ""
          if (!skills.includes(suggestion)) {
            skills.push(suggestion)
            renderSkills()
          }
          skillsSuggestions.innerHTML = ""
        })
        skillsSuggestions.appendChild(suggestionItem)
      })
    } else {
      skillsSuggestions.innerHTML = ""
    }
  })

  // Clear suggestions when clicking outside
  document.addEventListener("click", (e) => {
    if (!skillsInput.contains(e.target) && !skillsSuggestions.contains(e.target)) {
      skillsSuggestions.innerHTML = ""
    }
  })

  // Load job presets when the app starts
  loadJobPresets();
})

