// Ornithologist Edition - JavaScript Integration with Full Functionality
// Works with backend_direct.py for real model predictions

const API_URL = 'http://localhost:5000/api';

// Store prediction history
let predictionHistory = [];
const MAX_HISTORY = 5;

// Current prediction data
let currentPrediction = null;

// Charts for analysis tab
let frequencyChart = null;
let temporalChart = null;

// Scientific names mapping
const scientificNames = {
    "Barn Owl": "Tyto alba",
    "Black-headed Gull": "Chroicocephalus ridibundus", 
    "Blackcap": "Sylvia atricapilla",
    "Blue Tit": "Cyanistes caeruleus",
    "Bullfinch": "Pyrrhula pyrrhula",
    "Chaffinch": "Fringilla coelebs",
    "Chiffchaff": "Phylloscopus collybita",
    "Coal Tit": "Periparus ater",
    "Common Blackbird": "Turdus merula",
    "Coot": "Fulica atra",
    "Dunnock": "Prunella modularis",
    "Eurasian Magpie": "Pica pica",
    "Eurasian Wren": "Troglodytes troglodytes",
    "European Greenfinch": "Chloris chloris",
    "European Robin": "Erithacus rubecula",
    "Fieldfare": "Turdus pilaris",
    "Goldcrest": "Regulus regulus",
    "Great Spotted Woodpecker": "Dendrocopos major",
    "Great Tit": "Parus major",
    "House Sparrow": "Passer domesticus",
    "Jackdaw": "Corvus monedula",
    "Long-tailed Tit": "Aegithalos caudatus",
    "Mallard": "Anas platyrhynchos",
    "Moorhen": "Gallinula chloropus",
    "Nuthatch": "Sitta europaea",
    "Pied Wagtail": "Motacilla alba",
    "Starling": "Sturnus vulgaris",
    "Swallow": "Hirundo rustica",
    "Tawny Owl": "Strix aluco",
    "Water Rail": "Rallus aquaticus"
};

// Confusion species with likelihood percentages
const confusionSpecies = {
    "Coot": [{"species": "Moorhen", "likelihood": 85}, {"species": "Water Rail", "likelihood": 45}],
    "Moorhen": [{"species": "Coot", "likelihood": 85}, {"species": "Water Rail", "likelihood": 55}],
    "Blue Tit": [{"species": "Great Tit", "likelihood": 75}, {"species": "Coal Tit", "likelihood": 60}],
    "Great Tit": [{"species": "Blue Tit", "likelihood": 75}, {"species": "Coal Tit", "likelihood": 50}],
    "Common Blackbird": [{"species": "Fieldfare", "likelihood": 40}, {"species": "Starling", "likelihood": 35}],
    "European Robin": [{"species": "Dunnock", "likelihood": 65}, {"species": "Chaffinch", "likelihood": 30}],
    "House Sparrow": [{"species": "Dunnock", "likelihood": 55}, {"species": "Chaffinch", "likelihood": 40}],
    "Barn Owl": [{"species": "Tawny Owl", "likelihood": 60}],
    "Tawny Owl": [{"species": "Barn Owl", "likelihood": 60}]
};

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    
    // REMOVED: Don't load last active result on page load
    // loadLastActiveResult();
    
    // Populate species list
    populateSpeciesList();
});

function initializeApp() {
    // Hide loading overlay
    document.getElementById('loadingOverlay').style.display = 'none';
    
    // Initialize file upload
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const classifyBtn = document.getElementById('classifyBtn');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.background = 'rgba(74, 144, 226, 0.1)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.background = '';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.background = '';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    classifyBtn.addEventListener('click', classifyAudio);
    
    // Initialize action buttons
    document.getElementById('tryAnotherBtn').addEventListener('click', resetForm);
    document.getElementById('downloadReportBtn').addEventListener('click', showExportModal);
    
    // Initialize clear history button
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', clearPredictionHistory);
    }
    
    // Initialize field notes save button
    const saveFieldNotesBtn = document.getElementById('saveFieldNotesBtn');
    if (saveFieldNotesBtn) {
        saveFieldNotesBtn.addEventListener('click', saveFieldNotes);
    }
    
    // Initialize tabs
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            switchTab(tab.getAttribute('data-tab'));
        });
    });
    
    // Initialize export modal
    const closeModal = document.getElementById('closeModal');
    if (closeModal) {
        closeModal.addEventListener('click', () => {
            document.getElementById('exportModal').style.display = 'none';
        });
    }
    
    // Initialize export buttons
    initializeExportButtons();
    
    // Hide results section initially
    document.getElementById('resultsSection').style.display = 'none';
    
    // Load recent predictions from localStorage
    loadRecentPredictions();
    
    // Load and display saved field notes
    loadSavedFieldNotes();
}

// Initialize export functionality
function initializeExportButtons() {
    // BirdNET Export
    document.getElementById('exportBirdNET').addEventListener('click', () => {
        exportBirdNETFormat();
    });
    
    // Research JSON Export
    document.getElementById('exportResearch').addEventListener('click', () => {
        exportResearchJSON();
    });
    
    // eBird Export
    document.getElementById('exportEBird').addEventListener('click', () => {
        exportEBirdChecklist();
    });
    
    // PDF Export
    document.getElementById('exportPDF').addEventListener('click', () => {
        exportPDFReport();
    });
}

// Export functions
function exportBirdNETFormat() {
    if (!currentPrediction) return;
    
    // COMPREHENSIVE: Detailed BirdNET export with all analysis data
    const csv = [
        ['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Species', 'Common Name', 'Confidence', 'Agreement Score', 'Models Agreeing', 'Quality Score', 'SNR (dB)', 'Background Noise (%)', 'Peak Frequency (Hz)', 'Call Duration (s)', 'Syllable Count', 'Model Type', 'File', 'Processing Time (s)']
    ];
    
    // Get frequency range
    const freqRange = document.getElementById('freqRange').textContent;
    const freqMin = freqRange.split('-')[0] || '500';
    const freqMax = freqRange.split('-')[1] || '15000';
    
    // Add main consensus prediction
    csv.push([
        '1', 'Consensus', '1', '0.0', document.getElementById('callDuration').textContent || '3.0', 
        freqMin, freqMax,
        scientificNames[currentPrediction.predicted_species] || '',
        currentPrediction.predicted_species,
        currentPrediction.confidence.toFixed(2),
        currentPrediction.agreement_score.toFixed(2),
        currentPrediction.models_agreeing + '/' + currentPrediction.total_models,
        document.getElementById('qualityValue').textContent.replace('%', ''),
        document.getElementById('snrValue').textContent.replace(' dB', ''),
        document.getElementById('noiseValue').textContent.replace('%', ''),
        document.getElementById('peakFrequency').textContent,
        document.getElementById('callDuration').textContent,
        document.getElementById('syllableCount').textContent,
        'Ensemble Consensus',
        currentPrediction.filename || document.getElementById('fileName').textContent,
        currentPrediction.processingTime || ''
    ]);
    
    // Add individual model predictions with full details
    currentPrediction.model_predictions.forEach((pred, index) => {
        csv.push([
            (index + 2).toString(), 
            'Model ' + (index + 1), 
            '1', 
            '0.0', 
            document.getElementById('callDuration').textContent || '3.0',
            freqMin, 
            freqMax,
            scientificNames[pred.species] || '',
            pred.species,
            pred.confidence.toFixed(2),
            '', // Individual models don't have agreement score
            pred.species === currentPrediction.predicted_species ? 'Agrees' : 'Disagrees',
            '', // Quality score only for consensus
            '', // SNR only for consensus
            '', // Noise only for consensus
            '', // Peak freq only for consensus
            '', // Duration only for consensus
            '', // Syllables only for consensus
            pred.model,
            currentPrediction.filename || document.getElementById('fileName').textContent,
            ''
        ]);
    });
    
    // Add confusion species
    const confusedSpecies = confusionSpecies[currentPrediction.predicted_species] || [];
    confusedSpecies.forEach((item, index) => {
        csv.push([
            (currentPrediction.model_predictions.length + index + 3).toString(),
            'Confusion Species',
            '1',
            '0.0',
            '3.0',
            freqMin,
            freqMax,
            scientificNames[item.species] || '',
            item.species,
            item.likelihood.toString(),
            '',
            'Similar Species',
            '',
            '',
            '',
            '',
            '',
            '',
            'Confusion Likelihood',
            '',
            ''
        ]);
    });
    
    const csvContent = csv.map(row => row.join(',')).join('\n');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);
    downloadFile(csvContent, `birdnet_${currentPrediction.predicted_species.toLowerCase().replace(/ /g, '_')}_${timestamp}.csv`, 'text/csv');
    
    document.getElementById('exportModal').style.display = 'none';
}

function exportResearchJSON() {
    if (!currentPrediction) return;
    
    // COMPREHENSIVE: Export ALL available data
    const exportData = {
        metadata: {
            export_timestamp: new Date().toISOString(),
            system_version: '1.0.0',
            model_ensemble: '6-Model Consensus',
            location: 'UK',
            export_format: 'research_json_v3_complete'
        },
        file_info: {
            filename: currentPrediction.filename || document.getElementById('fileName').textContent,
            processing_time: currentPrediction.processingTime,
            analysis_timestamp: currentPrediction.time || document.getElementById('time').textContent
        },
        identification: {
            common_name: currentPrediction.predicted_species,
            scientific_name: scientificNames[currentPrediction.predicted_species],
            overall_confidence: currentPrediction.confidence,
            agreement_score: currentPrediction.agreement_score,
            models_in_agreement: currentPrediction.models_agreeing,
            total_models: currentPrediction.total_models,
            timestamp: currentPrediction.timestamp || new Date().toISOString()
        },
        individual_model_predictions: currentPrediction.model_predictions.map(pred => ({
            model_name: pred.model,
            model_type: pred.model,
            predicted_species: pred.species,
            scientific_name: scientificNames[pred.species] || 'Unknown',
            confidence_score: pred.confidence,
            matches_consensus: pred.species === currentPrediction.predicted_species,
            confidence_category: pred.confidence > 80 ? 'high' : pred.confidence > 60 ? 'medium' : 'low'
        })),
        model_agreement_matrix: generateAgreementMatrixData(currentPrediction.model_predictions),
        confusion_analysis: {
            similar_species: confusionSpecies[currentPrediction.predicted_species] || [],
            total_confusion_species: (confusionSpecies[currentPrediction.predicted_species] || []).length
        },
        audio_metrics: {
            call_duration_seconds: document.getElementById('callDuration').textContent,
            peak_frequency_hz: document.getElementById('peakFrequency').textContent,
            frequency_range_hz: document.getElementById('freqRange').textContent,
            frequency_min_hz: document.getElementById('freqRange').textContent.split('-')[0],
            frequency_max_hz: document.getElementById('freqRange').textContent.split('-')[1],
            syllable_count: document.getElementById('syllableCount').textContent,
            call_rate_per_minute: document.getElementById('callRate').textContent,
            inter_call_interval_seconds: document.getElementById('interCallInterval').textContent
        },
        recording_quality: {
            signal_noise_ratio_db: parseFloat(document.getElementById('snrValue').textContent),
            overall_quality_percent: parseFloat(document.getElementById('qualityValue').textContent),
            background_noise_percent: parseFloat(document.getElementById('noiseValue').textContent),
            quality_category: parseFloat(document.getElementById('qualityValue').textContent) > 80 ? 'excellent' : 
                            parseFloat(document.getElementById('qualityValue').textContent) > 60 ? 'good' : 
                            parseFloat(document.getElementById('qualityValue').textContent) > 40 ? 'fair' : 'poor'
        },
        spectrogram_data: {
            has_spectrogram: currentPrediction.spectrogram_base64 ? true : false,
            spectrogram_base64: currentPrediction.spectrogram_base64 || null
        },
        field_observations: getLatestFieldNote(),
        all_field_notes: JSON.parse(localStorage.getItem('fieldNotes') || '[]'),
        prediction_history: predictionHistory.map(h => ({
            species: h.species,
            scientific_name: scientificNames[h.species] || 'Unknown',
            confidence: h.confidence,
            file: h.file,
            time: h.time,
            processing_time: h.processingTime,
            agreement_score: h.agreement_score,
            models_agreeing: h.models_agreeing,
            total_models: h.total_models,
            timestamp: h.timestamp
        })),
        system_info: {
            total_predictions_made: predictionHistory.length,
            supported_species_count: Object.keys(scientificNames).length,
            supported_species_list: Object.keys(scientificNames).map(name => ({
                common_name: name,
                scientific_name: scientificNames[name]
            }))
        }
    };
    
    const jsonContent = JSON.stringify(exportData, null, 2);
    downloadFile(jsonContent, `bird_analysis_${currentPrediction.predicted_species.toLowerCase().replace(/ /g, '_')}_${Date.now()}.json`, 'application/json');
    
    document.getElementById('exportModal').style.display = 'none';
}

// Helper function to generate agreement matrix data
function generateAgreementMatrixData(predictions) {
    const matrix = [];
    predictions.forEach((pred1, i) => {
        const row = [];
        predictions.forEach((pred2, j) => {
            row.push({
                model1: pred1.model,
                model2: pred2.model,
                agrees: pred1.species === pred2.species,
                species1: pred1.species,
                species2: pred2.species
            });
        });
        matrix.push(row);
    });
    return matrix;
}

function exportEBirdChecklist() {
    if (!currentPrediction) return;
    
    const date = new Date();
    const checklist = `eBird Checklist
Date: ${date.toLocaleDateString()}
Time: ${date.toLocaleTimeString()}
Location: Field Location

Species Observed:
- ${currentPrediction.predicted_species} (${scientificNames[currentPrediction.predicted_species] || ''})
  Count: 1
  Confidence: ${currentPrediction.confidence.toFixed(1)}%
  Detection Method: Audio Recording
  
Notes: Identified using AI ensemble model with ${currentPrediction.models_agreeing}/${currentPrediction.total_models} models in agreement.`;
    
    downloadFile(checklist, 'ebird_checklist.txt', 'text/plain');
    
    document.getElementById('exportModal').style.display = 'none';
}

function exportPDFReport() {
    if (!currentPrediction) return;
    
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    
    // ENHANCED: More comprehensive PDF report
    // Title
    doc.setFontSize(22);
    doc.setTextColor(33, 150, 243);
    doc.text('Bird Species Identification Report', 105, 20, { align: 'center' });
    
    doc.setTextColor(0, 0, 0);
    doc.setFontSize(10);
    doc.text('AI Ensemble Analysis', 105, 28, { align: 'center' });
    
    // Date and metadata
    doc.setFontSize(10);
    doc.text(`Report Generated: ${new Date().toLocaleString()}`, 20, 40);
    doc.text(`System Version: 1.0.0 | 6-Model Ensemble`, 20, 46);
    
    // Draw line
    doc.setDrawColor(200, 200, 200);
    doc.line(20, 50, 190, 50);
    
    // Main Identification
    doc.setFontSize(16);
    doc.setTextColor(33, 150, 243);
    doc.text('Primary Identification', 20, 60);
    
    doc.setFontSize(12);
    doc.setTextColor(0, 0, 0);
    doc.text(`Species: ${currentPrediction.predicted_species}`, 20, 70);
    doc.text(`Scientific Name: ${scientificNames[currentPrediction.predicted_species] || 'N/A'}`, 20, 78);
    doc.setFontSize(14);
    doc.setTextColor(0, 128, 0);
    doc.text(`Confidence: ${currentPrediction.confidence.toFixed(1)}%`, 20, 88);
    
    // Consensus Metrics
    doc.setFontSize(12);
    doc.setTextColor(0, 0, 0);
    doc.text(`Agreement Score: ${currentPrediction.agreement_score.toFixed(1)}%`, 100, 70);
    doc.text(`Models in Agreement: ${currentPrediction.models_agreeing}/${currentPrediction.total_models}`, 100, 78);
    doc.text(`Processing Time: ${document.getElementById('processingTime').textContent}`, 100, 88);
    
    // Individual Model Predictions
    doc.setFontSize(14);
    doc.setTextColor(33, 150, 243);
    doc.text('Individual Model Analysis:', 20, 105);
    
    doc.setFontSize(10);
    doc.setTextColor(0, 0, 0);
    let yPos = 115;
    currentPrediction.model_predictions.forEach((pred, index) => {
        const color = pred.confidence > 80 ? [0, 128, 0] : pred.confidence > 60 ? [255, 152, 0] : [244, 67, 54];
        doc.setTextColor(...color);
        doc.text(`${index + 1}. ${pred.model}:`, 25, yPos);
        doc.setTextColor(0, 0, 0);
        doc.text(`${pred.species} (${pred.confidence.toFixed(1)}%)`, 85, yPos);
        yPos += 7;
    });
    
    // Audio Analysis Metrics
    yPos += 5;
    doc.setFontSize(14);
    doc.setTextColor(33, 150, 243);
    doc.text('Audio Analysis Metrics:', 20, yPos);
    
    doc.setFontSize(10);
    doc.setTextColor(0, 0, 0);
    yPos += 10;
    doc.text(`Call Duration: ${document.getElementById('callDuration').textContent} seconds`, 25, yPos);
    doc.text(`Peak Frequency: ${document.getElementById('peakFrequency').textContent} Hz`, 100, yPos);
    yPos += 7;
    doc.text(`Frequency Range: ${document.getElementById('freqRange').textContent} Hz`, 25, yPos);
    doc.text(`Syllable Count: ${document.getElementById('syllableCount').textContent}`, 100, yPos);
    yPos += 7;
    doc.text(`Call Rate: ${document.getElementById('callRate').textContent} per minute`, 25, yPos);
    doc.text(`Inter-call Interval: ${document.getElementById('interCallInterval').textContent} seconds`, 100, yPos);
    
    // Recording Quality
    yPos += 10;
    doc.setFontSize(14);
    doc.setTextColor(33, 150, 243);
    doc.text('Recording Quality Assessment:', 20, yPos);
    
    doc.setFontSize(10);
    doc.setTextColor(0, 0, 0);
    yPos += 10;
    doc.text(`Signal/Noise Ratio: ${document.getElementById('snrValue').textContent}`, 25, yPos);
    doc.text(`Overall Quality: ${document.getElementById('qualityValue').textContent}`, 80, yPos);
    doc.text(`Background Noise: ${document.getElementById('noiseValue').textContent}`, 135, yPos);
    
    // Confusion Species
    const confusedSpecies = confusionSpecies[currentPrediction.predicted_species] || [];
    if (confusedSpecies.length > 0) {
        yPos += 10;
        doc.setFontSize(14);
        doc.setTextColor(33, 150, 243);
        doc.text('Similar Species (Confusion Risk):', 20, yPos);
        
        doc.setFontSize(10);
        doc.setTextColor(0, 0, 0);
        yPos += 10;
        confusedSpecies.forEach(item => {
            doc.text(`• ${item.species} (${item.likelihood}% similarity)`, 25, yPos);
            yPos += 7;
        });
    }
    
    // Add page 2 for additional data
    if (yPos > 200) {
        doc.addPage();
        yPos = 20;
    }
    
    // Field Notes Summary
    const fieldNote = getLatestFieldNote();
    if (fieldNote) {
        doc.setFontSize(14);
        doc.setTextColor(33, 150, 243);
        doc.text('Field Observations:', 20, yPos);
        
        doc.setFontSize(10);
        doc.setTextColor(0, 0, 0);
        yPos += 10;
        
        if (fieldNote.location) {
            doc.text(`Location: ${fieldNote.location}`, 25, yPos);
            yPos += 7;
        }
        if (fieldNote.weather) {
            doc.text(`Weather: ${fieldNote.weather}`, 25, yPos);
            yPos += 7;
        }
        if (fieldNote.temperature) {
            doc.text(`Temperature: ${fieldNote.temperature}°C`, 25, yPos);
            yPos += 7;
        }
        if (fieldNote.habitat) {
            doc.text(`Habitat: ${fieldNote.habitat}`, 25, yPos);
            yPos += 7;
        }
    }
    
    // Analysis Summary
    yPos += 10;
    doc.setFontSize(14);
    doc.setTextColor(33, 150, 243);
    doc.text('Analysis Summary:', 20, yPos);
    
    doc.setFontSize(10);
    doc.setTextColor(0, 0, 0);
    yPos += 10;
    doc.text(`File: ${currentPrediction.filename || document.getElementById('fileName').textContent}`, 25, yPos);
    yPos += 7;
    doc.text(`Analysis Time: ${currentPrediction.time || document.getElementById('time').textContent}`, 25, yPos);
    yPos += 7;
    doc.text(`Total Models Used: ${currentPrediction.total_models}`, 25, yPos);
    yPos += 7;
    doc.text(`Models in Agreement: ${currentPrediction.models_agreeing}`, 25, yPos);
    
    // Footer
    doc.setFontSize(8);
    doc.setTextColor(128, 128, 128);
    doc.text('Generated by Bird Species Identification System', 105, 280, { align: 'center' });
    doc.text('Powered by 6-Model AI Ensemble: EfficientNet, ResNet-50, DenseNet, AST, ConvNeXt, PANNs', 105, 285, { align: 'center' });
    
    // Save PDF
    doc.save('bird_identification_report.pdf');
    
    document.getElementById('exportModal').style.display = 'none';
}

function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type: type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function getLatestFieldNote() {
    const notes = JSON.parse(localStorage.getItem('fieldNotes') || '[]');
    return notes.length > 0 ? notes[0] : null;
}

// Load and display last active result
function loadLastActiveResult() {
    const lastResult = localStorage.getItem('lastActiveResult');
    if (lastResult) {
        try {
            const result = JSON.parse(lastResult);
            displayStoredResult(result);
        } catch (error) {
            console.error('Error loading last active result:', error);
        }
    }
}

// Save current result as last active
function saveLastActiveResult(data, filename, processingTime) {
    const resultToStore = {
        predicted_species: data.predicted_species,
        confidence: data.confidence,
        model_predictions: data.model_predictions,
        agreement_score: data.agreement_score,
        models_agreeing: data.models_agreeing,
        total_models: data.total_models,
        filename: filename,
        processingTime: processingTime,
        time: new Date().toLocaleTimeString(),
        timestamp: Date.now(),
        // FIXED: Store audio metrics and spectrogram
        audio_metrics: data.audio_metrics || null,
        spectrogram_base64: data.spectrogram_base64 || null
    };
    
    localStorage.setItem('lastActiveResult', JSON.stringify(resultToStore));
}

// Display stored result
function displayStoredResult(result) {
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Update main prediction
    document.getElementById('speciesName').textContent = result.predicted_species;
    document.getElementById('scientificName').textContent = scientificNames[result.predicted_species] || '';
    document.getElementById('confidence').textContent = result.confidence.toFixed(1) + '%';
    document.getElementById('fileName').textContent = result.filename || '--';
    document.getElementById('time').textContent = result.time || new Date().toLocaleTimeString();
    document.getElementById('processingTime').textContent = result.processingTime ? result.processingTime + 's' : '--';
    
    // Store as current prediction WITH audio metrics
    currentPrediction = result;
    if (result.audio_metrics) {
        currentPrediction.audio_metrics = result.audio_metrics;
    }
    
    // Update bird image
    updateBirdImage(result.predicted_species);
    
    // Update confusion species
    updateConfusionSpecies(result.predicted_species);
    
    // Update agreement matrix
    if (result.model_predictions) {
        updateAgreementMatrix(result.model_predictions, result.predicted_species);
        updateModelPredictions(result.model_predictions);
    }
    
    // Update consensus stats
    document.getElementById('agreementScore').textContent = (result.agreement_score || 0).toFixed(0) + '%';
    document.getElementById('modelsAgreeing').textContent = (result.models_agreeing || 0) + '/' + (result.total_models || 6);
    document.getElementById('overallConfidence').textContent = (result.confidence || 0).toFixed(1) + '%';
    
    // FIXED: Update quality indicators with REAL data if available
    if (result.audio_metrics) {
        updateQualityIndicatorsReal(result.audio_metrics);
    } else {
        updateQualityIndicators();
    }
    
    // FIXED: Update analysis metrics with REAL data if available
    if (result.audio_metrics) {
        updateAnalysisMetricsReal(result.audio_metrics);
    } else {
        updateAnalysisMetrics();
    }
    
    // Enable action buttons
    document.getElementById('tryAnotherBtn').disabled = false;
    document.getElementById('downloadReportBtn').disabled = false;
    
    // Display spectrogram if available
    if (result.spectrogram_base64) {
        setTimeout(() => {
            displaySpectrogram(result.spectrogram_base64);
        }, 100);
    }
}

function handleFileSelect(file) {
    if (file && file.type.startsWith('audio/')) {
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.querySelector('.upload-text').textContent = file.name;
        document.getElementById('classifyBtn').disabled = false;
        currentFile = file;
    }
}

let currentFile = null;

async function classifyAudio() {
    if (!currentFile) return;
    
    const processingOverlay = document.getElementById('processingOverlay');
    const progressBar = document.getElementById('processingProgress');
    const progressText = document.getElementById('progressPercentage');
    const statusText = document.getElementById('processingStatus');
    
    processingOverlay.style.display = 'flex';
    
    // Animate progress
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        progressBar.style.width = progress + '%';
        progressText.textContent = Math.floor(progress) + '%';
        
        // Update status text
        if (progress < 30) {
            statusText.textContent = 'Uploading audio file...';
        } else if (progress < 60) {
            statusText.textContent = 'Analyzing frequency patterns...';
        } else {
            statusText.textContent = 'Running ensemble models...';
        }
    }, 500);
    
    const startTime = Date.now();
    
    try {
        const formData = new FormData();
        formData.append('audio', currentFile);
        
        const response = await fetch(`${API_URL}/classify`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        progressText.textContent = '100%';
        
        if (data.success) {
            const processingTime = ((Date.now() - startTime) / 1000).toFixed(1);
            displayResults(data, processingTime);
            
            // Add to history
            addToPredictionHistory(data, currentFile.name, processingTime);
            
            // Save as last active result
            saveLastActiveResult(data, currentFile.name, processingTime);
        } else {
            alert('Classification failed: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        clearInterval(progressInterval);
        alert('Error: ' + error.message);
    } finally {
        processingOverlay.style.display = 'none';
    }
}

function displayResults(data, processingTime) {
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Update main prediction
    document.getElementById('speciesName').textContent = data.predicted_species;
    document.getElementById('scientificName').textContent = scientificNames[data.predicted_species] || '';
    document.getElementById('confidence').textContent = data.confidence.toFixed(1) + '%';
    document.getElementById('fileName').textContent = currentFile ? currentFile.name : '--';
    document.getElementById('time').textContent = new Date().toLocaleTimeString();
    document.getElementById('processingTime').textContent = processingTime + 's';
    
    // Store current prediction FIRST (before updating image)
    currentPrediction = data;
    // Store audio metrics in currentPrediction for later use
    if (data.audio_metrics) {
        currentPrediction.audio_metrics = data.audio_metrics;
    }
    // Store original species for agreement calculation
    if (data.original_species) {
        currentPrediction.original_species = data.original_species;
    }
    
    // Update bird image (will check agreement score)
    updateBirdImage(data.predicted_species);
    
    // Add warning if low confidence or disagreement
    if (data.confidence < 50 || data.agreement_score < 50 || data.models_agreeing < 3) {
        // Update species name with warning
        const speciesElement = document.getElementById('speciesName');
        const originalText = speciesElement.textContent;
        speciesElement.innerHTML = originalText + '<span style="color: #f44336; font-size: 14px; display: block; margin-top: 10px;">⚠️ Low confidence - Result may be unreliable</span>';
    }
    
    // Check if species is out of scope or not in our supported list
    if (data.predicted_species === 'Unknown/Out of Scope' || !scientificNames.hasOwnProperty(data.predicted_species)) {
        document.getElementById('speciesName').innerHTML = 'Unknown/Out of Scope' + '<span style="color: #ffc107; font-size: 14px; display: block; margin-top: 10px;">⚠️ Species outside supported range</span>';
        document.getElementById('scientificName').textContent = 'Species not in UK database';
    }
    
    // Update confusion species
    updateConfusionSpecies(data.predicted_species);
    
    // Update agreement matrix - matrix shows model-to-model agreement
    updateAgreementMatrix(data.model_predictions);
    
    // Update consensus stats
    // Handle cases where we might have incomplete model data
    const actualModels = data.model_predictions ? data.model_predictions.length : 0;
    const expectedModels = data.total_models || 6;
    
    // Use the models_agreeing value from backend which uses original_species
    document.getElementById('agreementScore').textContent = data.agreement_score.toFixed(0) + '%';
    document.getElementById('modelsAgreeing').textContent = data.models_agreeing + '/' + data.total_models;
    document.getElementById('overallConfidence').textContent = data.confidence.toFixed(1) + '%';
    
    // Update individual model predictions
    updateModelPredictions(data.model_predictions);
    
    // FIXED: Update quality indicators with REAL data from backend
    if (data.audio_metrics) {
        updateQualityIndicatorsReal(data.audio_metrics);
    } else {
        updateQualityIndicators(); // Fallback to simulated
    }
    
    // FIXED: Update analysis metrics with REAL data from backend
    if (data.audio_metrics) {
        updateAnalysisMetricsReal(data.audio_metrics);
    } else {
        updateAnalysisMetrics(); // Fallback to simulated
    }
    
    // Update spectrogram if available
    if (data.spectrogram_base64) {
        displaySpectrogram(data.spectrogram_base64);
        // Store it in currentPrediction for later use
        currentPrediction.spectrogram_base64 = data.spectrogram_base64;
    }
    
    // Enable action buttons
    document.getElementById('tryAnotherBtn').disabled = false;
    document.getElementById('downloadReportBtn').disabled = false;
}

// MODIFIED: Update analysis metrics to check for real data first
function updateAnalysisMetrics() {
    // Check if we have real audio metrics stored
    if (currentPrediction && currentPrediction.audio_metrics) {
        updateAnalysisMetricsReal(currentPrediction.audio_metrics);
        return;
    }
    
    // Fallback to simulated data if no real metrics available
    console.warn('No real audio metrics available, using simulated data');
    
    const callDuration = (2.5 + Math.random() * 3).toFixed(2);
    const peakFreq = Math.floor(2000 + Math.random() * 6000);
    const minFreq = Math.floor(500 + Math.random() * 1500);
    const maxFreq = Math.floor(peakFreq + Math.random() * 2000);
    const syllables = Math.floor(3 + Math.random() * 8);
    const callRate = Math.floor(5 + Math.random() * 15);
    const interCall = (0.5 + Math.random() * 2).toFixed(2);
    
    // Update metric displays
    document.getElementById('callDuration').textContent = callDuration;
    document.getElementById('peakFrequency').textContent = peakFreq;
    document.getElementById('freqRange').textContent = `${minFreq}-${maxFreq}`;
    document.getElementById('syllableCount').textContent = syllables;
    document.getElementById('callRate').textContent = callRate;
    document.getElementById('interCallInterval').textContent = interCall;
    
    // Create frequency spectrum chart
    createFrequencyChart(minFreq, peakFreq, maxFreq);
    
    // Create temporal pattern chart
    createTemporalChart(parseFloat(callDuration), syllables);
}

// NEW: Update analysis metrics with REAL data from backend
function updateAnalysisMetricsReal(audioMetrics) {
    console.log('Using REAL audio metrics:', audioMetrics);
    console.log('Displaying real values - Call Duration:', audioMetrics.call_duration, 'Peak Freq:', audioMetrics.peak_frequency);
    
    // Update metric displays with REAL data
    document.getElementById('callDuration').textContent = audioMetrics.call_duration || '0.0';
    document.getElementById('peakFrequency').textContent = audioMetrics.peak_frequency || '0';
    document.getElementById('freqRange').textContent = `${audioMetrics.frequency_min || 0}-${audioMetrics.frequency_max || 0}`;
    document.getElementById('syllableCount').textContent = audioMetrics.syllable_count || '0';
    document.getElementById('callRate').textContent = audioMetrics.call_rate || '0';
    document.getElementById('interCallInterval').textContent = audioMetrics.inter_call_interval || '0.0';
    
    // Create frequency spectrum chart with REAL data
    if (audioMetrics.frequency_spectrum && audioMetrics.frequency_spectrum.frequencies && audioMetrics.frequency_spectrum.magnitudes) {
        createFrequencyChartReal(audioMetrics.frequency_spectrum);
    } else {
        // Fallback to generated chart if no spectrum data
        createFrequencyChart(audioMetrics.frequency_min || 1000, audioMetrics.peak_frequency || 3000, audioMetrics.frequency_max || 8000);
    }
    
    // Create temporal pattern chart with REAL data
    if (audioMetrics.temporal_pattern && audioMetrics.temporal_pattern.time_points && audioMetrics.temporal_pattern.amplitudes) {
        createTemporalChartReal(audioMetrics.temporal_pattern);
    } else {
        // Fallback to generated chart if no temporal data
        createTemporalChart(parseFloat(audioMetrics.call_duration || 2.5), audioMetrics.syllable_count || 5);
    }
}

// Create frequency spectrum chart
function createFrequencyChart(minFreq, peakFreq, maxFreq) {
    const ctx = document.getElementById('frequencyChart');
    if (!ctx) return;
    
    // FIXED: Destroy existing chart properly to prevent infinite growth
    if (frequencyChart) {
        frequencyChart.destroy();
        frequencyChart = null;
    }
    
    // Generate frequency data
    const frequencies = [];
    const amplitudes = [];
    for (let f = minFreq; f <= maxFreq; f += 100) {
        frequencies.push(f);
        // Create a peak around the peak frequency
        const distance = Math.abs(f - peakFreq);
        const amplitude = Math.max(0, 100 - (distance / 20));
        amplitudes.push(amplitude + Math.random() * 10);
    }
    
    frequencyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: frequencies,
            datasets: [{
                label: 'Amplitude (dB)',
                data: amplitudes,
                borderColor: '#4a90e2',
                backgroundColor: 'rgba(74, 144, 226, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,  // Allow flexible sizing
            aspectRatio: 2,  // Wider aspect ratio
            plugins: {
                legend: {
                    labels: { color: '#ffffff' }
                },
                title: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Frequency (Hz)',
                        color: '#ffffff'
                    },
                    ticks: { color: '#ffffff' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Amplitude (dB)',
                        color: '#ffffff'
                    },
                    ticks: { color: '#ffffff' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            }
        }
    });
}

// NEW: Create frequency spectrum chart with REAL data
function createFrequencyChartReal(frequencySpectrum) {
    const ctx = document.getElementById('frequencyChart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (frequencyChart) {
        frequencyChart.destroy();
        frequencyChart = null;
    }
    
    // Use real frequency and magnitude data
    const frequencies = frequencySpectrum.frequencies;
    const magnitudes = frequencySpectrum.magnitudes;
    
    // Limit data points for better visualization (every nth point)
    const maxPoints = 100;
    let step = Math.ceil(frequencies.length / maxPoints);
    const plottableFreqs = [];
    const plottableMags = [];
    
    for (let i = 0; i < frequencies.length; i += step) {
        plottableFreqs.push(Math.round(frequencies[i]));
        plottableMags.push(magnitudes[i]);
    }
    
    frequencyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: plottableFreqs,
            datasets: [{
                label: 'Amplitude (%)',
                data: plottableMags,
                borderColor: '#4a90e2',
                backgroundColor: 'rgba(74, 144, 226, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            aspectRatio: 2,
            plugins: {
                legend: {
                    labels: { color: '#ffffff' }
                },
                title: {
                    display: false  // Remove title
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Frequency (Hz)',
                        color: '#ffffff'
                    },
                    ticks: { 
                        color: '#ffffff',
                        maxTicksLimit: 10
                    },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Relative Amplitude (%)',
                        color: '#ffffff'
                    },
                    ticks: { color: '#ffffff' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            }
        }
    });
}

// Create temporal pattern chart
function createTemporalChart(duration, syllables) {
    const ctx = document.getElementById('temporalChart');
    if (!ctx) return;
    
    // FIXED: Destroy existing chart properly to prevent infinite growth
    if (temporalChart) {
        temporalChart.destroy();
        temporalChart = null;
    }
    
    // Generate temporal data (syllable pattern over time)
    const timePoints = [];
    const amplitudes = [];
    const numPoints = 50;
    const timeStep = duration / numPoints;
    
    for (let i = 0; i < numPoints; i++) {
        timePoints.push((i * timeStep).toFixed(2));
        // Create syllable peaks
        const syllablePosition = (i % Math.floor(numPoints / syllables)) === 0;
        amplitudes.push(syllablePosition ? 80 + Math.random() * 20 : Math.random() * 20);
    }
    
    temporalChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: timePoints,
            datasets: [{
                label: 'Sound Intensity',
                data: amplitudes,
                backgroundColor: 'rgba(76, 175, 80, 0.6)',
                borderColor: '#4CAF50',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#ffffff' }
                },
                title: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time (seconds)',
                        color: '#ffffff'
                    },
                    ticks: { 
                        color: '#ffffff',
                        maxTicksLimit: 10
                    },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Intensity',
                        color: '#ffffff'
                    },
                    ticks: { color: '#ffffff' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            }
        }
    });
}

// NEW: Create temporal pattern chart with REAL data
function createTemporalChartReal(temporalPattern) {
    const ctx = document.getElementById('temporalChart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (temporalChart) {
        temporalChart.destroy();
        temporalChart = null;
    }
    
    // Use real temporal data
    const timePoints = temporalPattern.time_points.map(t => t.toFixed(2));
    const amplitudes = temporalPattern.amplitudes;
    
    temporalChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timePoints,
            datasets: [{
                label: 'Sound Intensity',
                data: amplitudes,
                backgroundColor: 'rgba(76, 175, 80, 0.3)',
                borderColor: '#4CAF50',
                borderWidth: 2,
                fill: true,
                tension: 0.2,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#ffffff' }
                },
                title: {
                    display: false  // Remove title
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time (seconds)',
                        color: '#ffffff'
                    },
                    ticks: { 
                        color: '#ffffff',
                        maxTicksLimit: 20
                    },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Relative Intensity (%)',
                        color: '#ffffff'
                    },
                    ticks: { color: '#ffffff' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    min: 0,
                    max: 100
                }
            }
        }
    });
}

// Save field notes function
function saveFieldNotes() {
    const notes = {
        location: document.getElementById('location').value,
        weather: document.getElementById('weatherConditions').value,
        temperature: document.getElementById('temperature').value,
        windSpeed: document.getElementById('windSpeed').value,
        habitat: document.getElementById('habitatType').value,
        time: document.getElementById('timeOfDay').value,
        behavior: document.getElementById('behaviorNotes').value,
        additional: document.getElementById('additionalNotes').value,
        species: document.getElementById('speciesName').textContent,
        confidence: document.getElementById('confidence').textContent,
        timestamp: new Date().toISOString()
    };
    
    // Get existing notes or create new array
    let allNotes = JSON.parse(localStorage.getItem('fieldNotes') || '[]');
    
    // Add new note to beginning
    allNotes.unshift(notes);
    
    // Keep only last 10 notes
    if (allNotes.length > 10) {
        allNotes = allNotes.slice(0, 10);
    }
    
    // Save to localStorage
    localStorage.setItem('fieldNotes', JSON.stringify(allNotes));
    
    // Clear form
    document.getElementById('location').value = '';
    document.getElementById('weatherConditions').value = '';
    document.getElementById('temperature').value = '';
    document.getElementById('windSpeed').value = '';
    document.getElementById('habitatType').value = '';
    document.getElementById('timeOfDay').value = '';
    document.getElementById('behaviorNotes').value = '';
    document.getElementById('additionalNotes').value = '';
    
    // Reload display
    loadSavedFieldNotes();
    
    // Show success message
    alert('Field notes saved successfully!');
}

// Load and display saved field notes
function loadSavedFieldNotes() {
    const container = document.getElementById('savedNotesList');
    if (!container) return;
    
    const notes = JSON.parse(localStorage.getItem('fieldNotes') || '[]');
    
    if (notes.length === 0) {
        container.innerHTML = '<p style="color: rgba(255,255,255,0.6);">No saved field notes yet.</p>';
        return;
    }
    
    container.innerHTML = '';
    
    notes.forEach((note, index) => {
        const noteDiv = document.createElement('div');
        noteDiv.className = 'saved-note-item';
        
        const date = new Date(note.timestamp);
        const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
        
        noteDiv.innerHTML = `
            <h4>${note.species || 'Unknown Species'} - ${dateStr}</h4>
            <p><strong>Confidence:</strong> ${note.confidence || 'N/A'}</p>
            ${note.location ? `<p><strong>Location:</strong> ${note.location}</p>` : ''}
            ${note.weather ? `<p><strong>Weather:</strong> ${note.weather}</p>` : ''}
            ${note.temperature ? `<p><strong>Temperature:</strong> ${note.temperature}°C</p>` : ''}
            ${note.windSpeed ? `<p><strong>Wind Speed:</strong> ${note.windSpeed} mph</p>` : ''}
            ${note.habitat ? `<p><strong>Habitat:</strong> ${note.habitat}</p>` : ''}
            ${note.behavior ? `<p><strong>Behavior:</strong> ${note.behavior}</p>` : ''}
            ${note.additional ? `<p><strong>Notes:</strong> ${note.additional}</p>` : ''}
        `;
        
        container.appendChild(noteDiv);
    });
}

// Populate species list
function populateSpeciesList() {
    const speciesGrid = document.getElementById('speciesGrid');
    if (!speciesGrid) return;
    
    Object.keys(scientificNames).forEach(commonName => {
        const card = document.createElement('div');
        card.className = 'species-card';
        
        let imageName = commonName.toLowerCase().replace(/ /g, '_').replace('-', '_');
        let imageFile = imageName + '.jpg';
        
        card.innerHTML = `
            <img src="assets/birds/${imageFile}" alt="${commonName}" onerror="this.src='assets/birds/out_of_scope.png'">
            <div class="species-info">
                <div class="species-common-name">${commonName}</div>
                <div class="species-scientific-name">${scientificNames[commonName]}</div>
            </div>
        `;
        
        speciesGrid.appendChild(card);
    });
}

function updateBirdImage(species) {
    const birdImage = document.getElementById('birdImage');
    const imageLoader = document.getElementById('imageLoader');
    
    // Check if we should show out_of_scope image
    // Conditions: low confidence, low agreement, or unknown species
    const confidence = currentPrediction ? currentPrediction.confidence : 100;
    const agreementScore = currentPrediction ? currentPrediction.agreement_score : 100;
    const modelsAgreeing = currentPrediction ? currentPrediction.models_agreeing : 6;
    
    // Show out_of_scope if:
    // 1. Species is explicitly "Unknown/Out of Scope"
    // 2. Confidence is below 50%
    // 3. Agreement score is below 50% (STRICT CHECK)
    // 4. Less than 3 models agree (less than half)
    // 5. Species is not in our supported list
    const isOutOfScope = 
        species === 'Unknown/Out of Scope' ||
        confidence < 50 || 
        agreementScore < 50 || 
        modelsAgreeing < 3 ||
        !scientificNames.hasOwnProperty(species);
    
    if (isOutOfScope) {
        // Use out_of_scope image
        birdImage.src = 'assets/birds/out_of_scope.png';
        imageLoader.style.display = 'none';
        birdImage.style.display = 'block';
        
        // Log reason for debugging
        console.log('Showing out_of_scope image - Reasons:');
        if (confidence < 50) console.log('  - Low confidence:', confidence + '%');
        if (agreementScore < 50) console.log('  - Low agreement:', agreementScore + '%');
        if (modelsAgreeing < 3) console.log('  - Few models agree:', modelsAgreeing);
        if (!scientificNames.hasOwnProperty(species)) console.log('  - Unknown species:', species);
        
        return;
    }
    
    // Convert species name to filename format
    let filename = species.toLowerCase().replace(/ /g, '_').replace(/-/g, '_');
    filename = filename + '.jpg';
    
    const imagePath = `assets/birds/${filename}`;
    
    imageLoader.style.display = 'block';
    birdImage.style.display = 'none';
    
    // Create new image to test loading
    const img = new Image();
    img.onload = function() {
        birdImage.src = this.src;
        imageLoader.style.display = 'none';
        birdImage.style.display = 'block';
    };
    img.onerror = function() {
        birdImage.src = 'assets/birds/out_of_scope.png';
        imageLoader.style.display = 'none';
        birdImage.style.display = 'block';
    };
    img.src = imagePath;
}

function updateConfusionSpecies(species) {
    const container = document.getElementById('confusionSpecies');
    container.innerHTML = '';
    
    // If species is out of scope, show appropriate message
    if (species === 'Unknown/Out of Scope') {
        container.innerHTML = '<p style="color: rgba(255,255,255,0.5);">Unable to determine similar species for out-of-scope identification</p>';
        return;
    }
    
    const similar = confusionSpecies[species] || [];
    similar.forEach(item => {
        const div = document.createElement('div');
        div.className = 'confusion-item';
        div.innerHTML = `${item.species} <span class="confusion-percentage">${item.likelihood}%</span>`;
        div.onclick = () => showSpeciesComparison(species, item.species);
        container.appendChild(div);
    });
    
    if (similar.length === 0) {
        container.innerHTML = '<p style="color: rgba(255,255,255,0.5);">No commonly confused species</p>';
    }
}

function updateAgreementMatrix(predictions) {
    const matrix = document.getElementById('agreementMatrix');
    matrix.innerHTML = '';
    
    // Check if we have valid predictions
    if (!predictions || predictions.length === 0) {
        matrix.innerHTML = '<div style="padding: 20px; text-align: center; color: rgba(255,255,255,0.6);">No model data available</div>';
        return;
    }
    
    // If we only have one model, show a simplified view
    if (predictions.length === 1) {
        matrix.innerHTML = `
            <div style="padding: 20px; text-align: center; color: rgba(255,255,255,0.8);">
                <p>Only one model provided predictions:</p>
                <p style="margin-top: 10px;"><strong>${predictions[0].model}:</strong> ${predictions[0].species} (${predictions[0].confidence.toFixed(1)}%)</p>
                <p style="margin-top: 15px; color: rgba(255,255,255,0.6); font-size: 12px;">Note: Full ensemble analysis unavailable</p>
            </div>
        `;
        return;
    }
    
    // REDESIGNED: Better table structure for agreement matrix
    // Create header row
    const headerRow = document.createElement('div');
    headerRow.className = 'matrix-row';
    
    // Corner cell
    const cornerCell = document.createElement('div');
    cornerCell.className = 'matrix-cell matrix-corner';
    cornerCell.textContent = 'Models';
    headerRow.appendChild(cornerCell);
    
    // Model abbreviations for column headers
    const modelAbbreviations = [
        'EN-B1',  // EfficientNet-B1
        'RN-50',  // ResNet-50 CBAM
        'DN-121', // DenseNet-121
        'AST',    // Audio Spectrogram Transformer
        'CNX-T',  // ConvNeXt-Tiny
        'PANNs'   // PANNs CNN14
    ];
    
    predictions.forEach((pred, i) => {
        const cell = document.createElement('div');
        cell.className = 'matrix-cell matrix-header-top';
        cell.textContent = modelAbbreviations[i] || 'M' + (i+1);
        cell.title = pred.model;
        headerRow.appendChild(cell);
    });
    
    matrix.appendChild(headerRow);
    
    // Create data rows
    predictions.forEach((pred, i) => {
        const row = document.createElement('div');
        row.className = 'matrix-row';
        
        // Row header with full model name
        const rowHeader = document.createElement('div');
        rowHeader.className = 'matrix-cell matrix-header-side';
        rowHeader.textContent = pred.model;
        row.appendChild(rowHeader);
        
        // Agreement cells
        predictions.forEach((pred2, j) => {
            const cell = document.createElement('div');
            const agrees = pred.species === pred2.species;
            cell.className = `matrix-cell ${agrees ? 'agrees' : 'disagrees'}`;
            cell.textContent = agrees ? '✓' : '✗';
            cell.title = `${pred.model} vs ${pred2.model}: ${agrees ? 'Agree' : 'Disagree'}`;
            row.appendChild(cell);
        });
        
        matrix.appendChild(row);
    });
}

// FIXED: Update model predictions with centered species names and confidence label
function updateModelPredictions(predictions) {
    const container = document.getElementById('modelsList');
    container.innerHTML = '';
    
    predictions.forEach(pred => {
        const div = document.createElement('div');
        div.className = 'model-prediction-item';
        
        // Determine confidence level for styling
        let confidenceClass = 'confidence-low';
        if (pred.confidence > 80) {
            confidenceClass = 'confidence-high';
        } else if (pred.confidence > 60) {
            confidenceClass = 'confidence-medium';
        }
        
        div.innerHTML = `
            <div class="model-info">
                <span class="model-name">${pred.model}</span>
                <span class="predicted-species">${pred.species}</span>
            </div>
            <div class="model-confidence ${confidenceClass}">
                <span class="confidence-label">Confidence</span>
                <span>${pred.confidence.toFixed(1)}%</span>
            </div>
        `;
        
        container.appendChild(div);
    });
}

// MODIFIED: Update quality indicators to check for real data first
function updateQualityIndicators() {
    // Check if we have real audio metrics stored
    if (currentPrediction && currentPrediction.audio_metrics) {
        updateQualityIndicatorsReal(currentPrediction.audio_metrics);
        return;
    }
    
    // Fallback to simulated data
    console.warn('No real quality metrics available, using simulated data');
    
    const snr = 60 + Math.random() * 40;
    const quality = 50 + Math.random() * 50;
    const noise = 100 - quality;
    
    // Map SNR to percentage
    const snrPercent = Math.min(100, snr);
    
    // Update bars
    document.getElementById('snrBar').style.width = snrPercent + '%';
    document.getElementById('qualityBar').style.width = quality + '%';
    document.getElementById('noiseBar').style.width = noise + '%';
    
    // Update bar colors
    const snrBar = document.getElementById('snrBar');
    const qualityBar = document.getElementById('qualityBar');
    const noiseBar = document.getElementById('noiseBar');
    
    // Color code SNR bar (high is good)
    if (snr >= 70) {
        snrBar.style.background = '#4CAF50';  // Green
    } else if (snr >= 40) {
        snrBar.style.background = '#FFC107';  // Yellow/orange
    } else {
        snrBar.style.background = '#f44336';  // Red
    }
    
    // Color code quality bar (high is good)
    if (quality >= 80) {
        qualityBar.style.background = '#4CAF50';  // Green
    } else if (quality >= 50) {
        qualityBar.style.background = '#FFC107';  // Yellow/orange
    } else {
        qualityBar.style.background = '#f44336';  // Red
    }
    
    // Color code noise bar (LOW is good)
    if (noise <= 20) {
        noiseBar.style.background = '#4CAF50';  // Green
    } else if (noise <= 50) {
        noiseBar.style.background = '#FFC107';  // Yellow/orange
    } else {
        noiseBar.style.background = '#f44336';  // Red
    }
    
    // Update values with proper formatting
    document.getElementById('snrValue').innerHTML = `<span class="quality-value">${snr.toFixed(0)} dB</span>`;
    document.getElementById('qualityValue').innerHTML = `<span class="quality-value">${quality.toFixed(0)}%</span>`;
    document.getElementById('noiseValue').innerHTML = `<span class="quality-value">${noise.toFixed(0)}%</span>`;
}

// NEW: Update quality indicators with REAL data from backend
function updateQualityIndicatorsReal(audioMetrics) {
    console.log('Using REAL quality metrics:', {
        snr: audioMetrics.snr_db,
        quality: audioMetrics.quality_percent,
        noise: audioMetrics.background_noise
    });
    
    // Use REAL SNR, quality, and noise values from backend
    const snr = audioMetrics.snr_db || 0;
    const quality = audioMetrics.quality_percent || 0;
    const noise = audioMetrics.background_noise || 0;
    
    // Map SNR to percentage (0-100 dB range)
    const snrPercent = Math.min(100, snr);
    
    // Update bars
    document.getElementById('snrBar').style.width = snrPercent + '%';
    document.getElementById('qualityBar').style.width = quality + '%';
    document.getElementById('noiseBar').style.width = noise + '%';
    
    // Update bar colors based on values
    const snrBar = document.getElementById('snrBar');
    const qualityBar = document.getElementById('qualityBar');
    const noiseBar = document.getElementById('noiseBar');
    
    // Color code SNR bar (high is good)
    if (snr >= 70) {
        snrBar.style.background = '#4CAF50';  // Green for good
    } else if (snr >= 40) {
        snrBar.style.background = '#FFC107';  // Yellow/orange for medium
    } else {
        snrBar.style.background = '#f44336';  // Red for poor
    }
    
    // Color code quality bar (high is good)
    if (quality >= 80) {
        qualityBar.style.background = '#4CAF50';  // Green for excellent
    } else if (quality >= 50) {
        qualityBar.style.background = '#FFC107';  // Yellow/orange for medium
    } else {
        qualityBar.style.background = '#f44336';  // Red for poor
    }
    
    // Color code noise bar (LOW is good, HIGH is bad)
    if (noise <= 20) {
        noiseBar.style.background = '#4CAF50';  // Green for low noise (good)
    } else if (noise <= 50) {
        noiseBar.style.background = '#FFC107';  // Yellow/orange for medium noise
    } else {
        noiseBar.style.background = '#f44336';  // Red for high noise (bad)
    }
    
    // Update values with proper formatting
    document.getElementById('snrValue').innerHTML = `<span class="quality-value">${snr.toFixed(1)} dB</span>`;
    document.getElementById('qualityValue').innerHTML = `<span class="quality-value">${quality.toFixed(1)}%</span>`;
    document.getElementById('noiseValue').innerHTML = `<span class="quality-value">${noise.toFixed(1)}%</span>`;
}

function displaySpectrogram(base64Data) {
    const canvas = document.getElementById('spectrogramCanvas');
    const placeholder = document.getElementById('spectrogramPlaceholder');
    
    if (base64Data) {
        const img = new Image();
        img.onload = function() {
            // ENHANCED: High-resolution spectrogram rendering
            const ctx = canvas.getContext('2d');
            
            // Get container dimensions - use full width
            const container = canvas.parentElement;
            const containerStyle = window.getComputedStyle(container);
            const containerPadding = parseFloat(containerStyle.paddingLeft) + parseFloat(containerStyle.paddingRight);
            const maxWidth = container.clientWidth - containerPadding;
            const maxHeight = 400; // Increased height for better visibility
            
            // Calculate optimal dimensions - always use full width
            const imgAspectRatio = img.width / img.height;
            let canvasWidth, canvasHeight;
            
            // Always use full available width
            canvasWidth = maxWidth;
            canvasHeight = canvasWidth / imgAspectRatio;
            
            // If height is too tall, constrain by height instead
            if (canvasHeight > maxHeight) {
                canvasHeight = maxHeight;
                canvasWidth = maxHeight * imgAspectRatio;
            }
            
            // Ensure minimum dimensions
            canvasWidth = Math.max(canvasWidth, maxWidth * 0.95); // Use at least 95% of available width
            canvasHeight = Math.max(canvasHeight, 250);
            
            // Set canvas size with device pixel ratio for high DPI displays
            const dpr = window.devicePixelRatio || 1;
            
            // Set the actual dimensions of the canvas
            canvas.width = canvasWidth * dpr;
            canvas.height = canvasHeight * dpr;
            
            // Set the display size (CSS pixels)
            canvas.style.width = canvasWidth + 'px';
            canvas.style.height = canvasHeight + 'px';
            
            // Scale the drawing context to match device pixel ratio
            ctx.scale(dpr, dpr);
            
            // Use high quality image rendering
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            
            // Clear canvas first
            ctx.clearRect(0, 0, canvasWidth, canvasHeight);
            
            // Draw image at the calculated size
            ctx.drawImage(img, 0, 0, canvasWidth, canvasHeight);
            
            // Hide placeholder and show canvas
            placeholder.style.display = 'none';
            canvas.style.display = 'block';
            
            // Store spectrogram data for export
            if (currentPrediction) {
                currentPrediction.spectrogram_base64 = base64Data;
            }
        };
        img.onerror = function() {
            console.error('Failed to load spectrogram');
            placeholder.innerHTML = '<p>Spectrogram generation failed</p>';
        };
        img.src = 'data:image/png;base64,' + base64Data;
    } else {
        placeholder.innerHTML = '<p>No spectrogram data available</p>';
    }
}

function addToPredictionHistory(data, filename, processingTime) {
    const prediction = {
        species: data.predicted_species,
        confidence: data.confidence.toFixed(1),
        file: filename,
        time: new Date().toLocaleTimeString(),
        processingTime: processingTime,
        modelPredictions: data.model_predictions,
        agreement_score: data.agreement_score,
        models_agreeing: data.models_agreeing,
        total_models: data.total_models,
        timestamp: Date.now(),
        // FIXED: Store audio metrics in prediction history
        audio_metrics: data.audio_metrics || null,
        spectrogram_base64: data.spectrogram_base64 || null
    };
    
    predictionHistory.unshift(prediction);
    if (predictionHistory.length > MAX_HISTORY) {
        predictionHistory = predictionHistory.slice(0, MAX_HISTORY);
    }
    
    // Save to localStorage
    localStorage.setItem('predictionHistory', JSON.stringify(predictionHistory));
    
    updateRecentPredictions();
}

function updateRecentPredictions() {
    const container = document.getElementById('recentPredictions');
    container.innerHTML = '';
    
    predictionHistory.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'recent-prediction-item';
        item.style.cssText = 'padding: 15px; margin-bottom: 10px; border-radius: 8px; background: rgba(255,255,255,0.05); cursor: pointer;';
        
        item.innerHTML = `
            <div style="color: #ffffff;"><strong style="color: #64b5f6;">Species:</strong> ${pred.species}</div>
            <div style="color: #ffffff;"><strong style="color: #64b5f6;">File:</strong> ${pred.file}</div>
            <div style="color: #ffffff;"><strong style="color: #64b5f6;">Time:</strong> ${pred.time}</div>
            <div style="color: #ffffff;"><strong style="color: #64b5f6;">Confidence:</strong> ${pred.confidence}%</div>
        `;
        
        item.onclick = () => {
            console.log(`Loading prediction ${index}`);
            loadPrediction(index);
        };
        container.appendChild(item);
    });
}

function loadPrediction(index) {
    const pred = predictionHistory[index];
    if (pred) {
        displayStoredResult({
            predicted_species: pred.species,
            confidence: parseFloat(pred.confidence),
            model_predictions: pred.modelPredictions || [],
            agreement_score: pred.agreement_score || 80,
            models_agreeing: pred.models_agreeing || 5,
            total_models: pred.total_models || 6,
            filename: pred.file,
            processingTime: pred.processingTime,
            time: pred.time,
            // FIXED: Pass audio metrics and spectrogram from history
            audio_metrics: pred.audio_metrics || null,
            spectrogram_base64: pred.spectrogram_base64 || null
        });
    }
}

function loadRecentPredictions() {
    const saved = localStorage.getItem('predictionHistory');
    if (saved) {
        predictionHistory = JSON.parse(saved);
        // Log what was loaded from localStorage
        console.log('Loaded prediction history from localStorage:');
        predictionHistory.forEach((pred, i) => {
            console.log(`  ${i}: ${pred.species} - Has metrics: ${pred.audio_metrics ? 'YES' : 'NO'}`);
        });
        updateRecentPredictions();
    }
}

// Clear prediction history
function clearPredictionHistory() {
    if (confirm('Are you sure you want to clear all prediction history?')) {
        predictionHistory = [];
        localStorage.removeItem('predictionHistory');
        updateRecentPredictions();
        console.log('Prediction history cleared');
        
        // Show a temporary message
        const container = document.getElementById('recentPredictions');
        container.innerHTML = '<p style="color: rgba(255,255,255,0.6); padding: 15px; text-align: center;">History cleared</p>';
        
        // After 2 seconds, show the empty state
        setTimeout(() => {
            if (predictionHistory.length === 0) {
                container.innerHTML = '<p style="color: rgba(255,255,255,0.6); padding: 15px; text-align: center;">No recent predictions</p>';
            }
        }, 2000);
    }
}

function resetForm() {
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('uploadArea').querySelector('.upload-text').textContent = 'Drop Audio File Here (WAV/MP3) or Click to Browse';
    document.getElementById('classifyBtn').disabled = true;
    currentFile = null;
    currentPrediction = null;
}

function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none';
    });
    
    // Show selected tab
    const selectedTab = document.getElementById(tabName + 'Tab');
    if (selectedTab) {
        selectedTab.style.display = 'block';
        
        // If switching to analysis tab, update the charts and spectrogram
        if (tabName === 'analysis') {
            // Use real metrics if available
            if (currentPrediction && currentPrediction.audio_metrics) {
                updateAnalysisMetricsReal(currentPrediction.audio_metrics);
            } else {
                updateAnalysisMetrics();
            }
            
            // Re-display spectrogram if available
            if (currentPrediction && currentPrediction.spectrogram_base64) {
                setTimeout(() => {
                    displaySpectrogram(currentPrediction.spectrogram_base64);
                }, 100);
            }
        }
    }
    
    // Update active class
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelector(`.nav-tab[data-tab="${tabName}"]`).classList.add('active');
}

function showExportModal() {
    document.getElementById('exportModal').style.display = 'block';
}

function showSpeciesComparison(species1, species2) {
    // This would show a detailed comparison between two species
    console.log(`Compare ${species1} with ${species2}`);
}

// Export the app object for external access
window.birdApp = {
    classifyAudio,
    resetForm,
    currentPrediction,
    predictionHistory,
    saveFieldNotes,
    updateAnalysisMetrics
};
