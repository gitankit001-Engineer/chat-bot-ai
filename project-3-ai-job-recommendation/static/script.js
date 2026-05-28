async function getRecommendations() {
    const experience = document.getElementById('experience').value;
    const s1 = document.getElementById('s1').value;
    const s2 = document.getElementById('s2').value;
    const s3 = document.getElementById('s3').value;
    
    // UI Loading state
    document.getElementById('loader').style.display = 'block';
    document.getElementById('results').innerHTML = '';

    try {
        const response = await fetch('/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                skills: [s1, s2, s3],
                experience: experience 
            })
        });
        
        if (!response.ok) throw new Error("Server communication failed.");
        
        const data = await response.json();
        document.getElementById('loader').style.display = 'none';
        
        let html = '';
        data.forEach((item, index) => {
            html += `
            <div class="result-card">
                <h3>Rank #${index + 1} | Job ID: ${item.job_id}</h3>
                <div class="match-stat">${item.match}%</div>
                <p class="stack-list"><strong>Required Stack:</strong><br>${item.stack}</p>
            </div>`;
        });
        document.getElementById('results').innerHTML = html;

    } catch (error) {
        document.getElementById('loader').style.display = 'none';
        document.getElementById('results').innerHTML = `
            <div class="result-card" style="border-left-color: #ff3366;">
                <h3 style="color: #ff3366;">System Error</h3>
                <p class="stack-list">Failed to generate AI predictions. Please try again.</p>
            </div>`;
    }
}