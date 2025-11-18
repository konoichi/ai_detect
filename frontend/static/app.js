let API_BASE_URL = null;

async function loadConfig(){
    const cfg = await fetch('config.json').then(r=>r.json());
    API_BASE_URL = cfg.API_BASE_URL;
}
loadConfig();

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const urlInput = document.getElementById("urlInput");
const results = document.getElementById("results");

dropzone.addEventListener("click", ()=> fileInput.click());
dropzone.addEventListener("dragover", e=>{ e.preventDefault(); dropzone.classList.add("drag-over"); });
dropzone.addEventListener("dragleave", ()=> dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop", e=>{
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    fileInput.files = e.dataTransfer.files;
    previewAndAnalyze(file);
});

fileInput.addEventListener("change", ()=> {
    const file = fileInput.files[0];
    previewAndAnalyze(file);
});

analyzeBtn.addEventListener("click", ()=> {
    const url = urlInput.value.trim();
    if(url) fetchImageFromURL(url);
});

async function fetchImageFromURL(url){
    const blob = await fetch(url).then(r=>r.blob());
    previewAndAnalyze(new File([blob], "remote.jpg"));
}

async function previewAndAnalyze(file){
    showPreview(file);
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(API_BASE_URL + "/detect/image", {
        method: "POST",
        body: fd
    });
    const data = await r.json();
    showResults(data);
}

function showPreview(file){
    const imgURL = URL.createObjectURL(file);
    results.innerHTML = `<div class="stat-card"><h3>Preview</h3><img style="max-width:100%;" src="${imgURL}"></div>`;
}

function showResults(data){
    const w = data.warnings?.map(x=>`<li>${x}</li>`).join("") || "";
    const score = data.is_ai_probability;

    results.innerHTML += `
    <div class="stat-card"><h3>KI-Wahrscheinlichkeit</h3><div class="value">${(score*100).toFixed(1)}%</div></div>
    <div class="stat-card"><h3>Warnungen</h3><ul>${w}</ul></div>
    <div class="stat-card"><h3>Details</h3><pre>${JSON.stringify(data.dimensions,null,2)}</pre></div>
    `;
}
