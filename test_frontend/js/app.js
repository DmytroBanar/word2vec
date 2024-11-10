// Similarity function for finding similar words
async function findSimilarWords() {
    try {
        // Retrieve the input word
        const word = document.getElementById("wordInput").value;

        // Define the URL with parameters
        const url = `http://127.0.0.1:5000/similarity?word=${encodeURIComponent(word)}&num_neighbors=5`;

        // Fetch data from the API
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        // Check if response is successful
        if (!response.ok) {
            throw new Error("Word not found in model.");
        }

        // Parse JSON response
        const data = await response.json();

        // Handle the response data
        displayResults(data);
        console.log(data);

    } catch (error) {
        console.error("Error:", error);
        alert(error.message);
    }
}

// Display results for similarity
function displayResults(data) {
    const resultElement = document.getElementById("result");
    resultElement.innerHTML = "";
    data.forEach(([word, similarity]) => {
        const li = document.createElement("li");
        li.textContent = `${word}: ${similarity}`;
        resultElement.appendChild(li);
    });
}

// Analogy function for finding analogy words
async function findAnalogies() {
    // Get input values
    const word = document.getElementById('wordInputAnalogy').value;
    const pair = document.getElementById('pairInput').value.split(',').map(s => s.trim());
    const numAnalogies = document.getElementById('numAnalogiesInput').value || 10;

    // Prepare payload
    const payload = {
        word: word,
        pair: pair,
        num_analogies: parseInt(numAnalogies, 10)
    };

    try {
        // Send POST request to /analogy endpoint
        const response = await fetch('http://127.0.0.1:5000/antology', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error(`Error: ${response.status}`);

        const data = await response.json();
        displayAnalogyResults(data);
    } catch (error) {
        console.error("Request Error:", error);
    }
}

// Display results for analogies
function displayAnalogyResults(data) {
    const analogyResultList = document.getElementById('analogyResult');
    analogyResultList.innerHTML = '';  // Clear previous results

    data.forEach(item => {
        const li = document.createElement('li');
        li.textContent = `${item[0]} (Similarity: ${item[1]})`;
        analogyResultList.appendChild(li);
    });
}

async function plotClusters() {
    const wordsInput = document.getElementById('wordsInput').value.split(',').map(s => s.trim());
    const numClusters = 5;

    try {
        const response = await fetch('http://127.0.0.1:5000/plot_clusters', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ words: wordsInput, num_clusters: numClusters })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || "Failed to generate clusters.");
        }

        const data = await response.json();
        if (data.image) {
            document.getElementById('clusterImage').src = 'data:image/png;base64,' + data.image;
        } else {
            console.error(data.error);
            alert("Error: " + data.error);
        }
    } catch (error) {
        console.error("Error:", error.message);
        alert("Error: " + error.message);
    }
}
