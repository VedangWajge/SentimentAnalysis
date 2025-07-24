import React, { useState } from 'react';
import {
  Activity,
  FileText,
  Keyboard
} from 'lucide-react';
import { Bar } from 'react-chartjs-2';
import { ChartData } from 'chart.js';
import 'chart.js/auto';

const App = () => {
  const [inputType, setInputType] = useState('text');
  const [textInput, setTextInput] = useState('');
  const [fileContent, setFileContent] = useState('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const fetchSentiment = async (content: string) => {
    setLoading(true);
    try {
      const res = await fetch('https://sentimentanalysisbackend-kog3.onrender.com/analyze-comparison', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputType === 'text' ? content : '',
          fileContent: inputType === 'file' ? content : '',
        })
      });
      const data = await res.json();
      if (Array.isArray(data)) {
        setResult(data[0]);
      } else {
        setResult(data);
      }
    } catch (err) {
      console.error('Sentiment analysis failed:', err);
    }
    setLoading(false);
  };

  const handleAnalyze = () => {
    let content = '';
    if (inputType === 'text') content = textInput;
    if (inputType === 'file') content = fileContent;
    if (content.trim()) fetchSentiment(content);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const raw = event.target?.result as string;
        const truncated = raw.split(" ").slice(0, 150).join(" ");
        setFileContent(truncated);
      };
      reader.readAsText(file);
    }
  };

  const chartData: ChartData<'bar', number[], string> | null = result ? {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [
      {
        label: 'VADER',
        data: [
          result.vader.pos * 100,
          result.vader.neu * 100,
          result.vader.neg * 100
        ],
        backgroundColor: 'rgba(59, 130, 246, 0.6)'
      },
      {
        label: 'Hugging Face',
        data: [
          result.huggingface.sentiment === 'Positive' ? 100 : 0,
          result.huggingface.sentiment === 'Neutral' ? 100 : 0,
          result.huggingface.sentiment === 'Negative' ? 100 : 0,
        ],
        backgroundColor: 'rgba(139, 92, 246, 0.6)'
      }
    ]
  } : null;

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 h-16 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <Activity className="w-6 h-6 text-blue-600" />
            <span className="text-xl font-bold text-gray-800">SentimentScope</span>
          </div>
        </div>
      </nav>

      <main className="max-w-5xl mx-auto px-4 py-10 space-y-8">
        <h1 className="text-2xl font-bold">Compare Sentiment Analysis with VADER & Hugging Face</h1>

        <div className="flex gap-4">
          <button onClick={() => setInputType('text')} className={`px-4 py-2 rounded ${inputType === 'text' ? 'bg-blue-600 text-white' : 'bg-white border'}`}><Keyboard className="inline w-4 mr-1" /> Text</button>
          <button onClick={() => setInputType('file')} className={`px-4 py-2 rounded ${inputType === 'file' ? 'bg-blue-600 text-white' : 'bg-white border'}`}><FileText className="inline w-4 mr-1" /> File</button>
        </div>

        {inputType === 'text' && (
          <textarea value={textInput} onChange={e => setTextInput(e.target.value)} className="w-full h-32 p-4 border rounded" placeholder="Enter text..." />
        )}

        {inputType === 'file' && (
          <input type="file" accept=".txt" onChange={handleFileUpload} />
        )}

        <button
          onClick={handleAnalyze}
          className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
          disabled={loading || (!textInput && !fileContent)}
        >
          {loading ? 'Analyzing...' : 'Analyze Sentiment'}
        </button>

        {result && (
          <div className="space-y-6">
            {chartData && <Bar data={chartData} />}

            <div className="bg-gray-100 p-4 border rounded">
              <h3 className="font-semibold mb-2 text-gray-700">Analyzed Text:</h3>
              <p className="text-sm text-gray-700 whitespace-pre-line">{result.input}</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* VADER */}
              <div className="bg-white p-6 border rounded">
                <h3 className="text-blue-700 font-semibold mb-2">VADER Results</h3>
                <p>Positive: {(result.vader.pos * 100).toFixed(2)}%</p>
                <p>Neutral: {(result.vader.neu * 100).toFixed(2)}%</p>
                <p>Negative: {(result.vader.neg * 100).toFixed(2)}%</p>
                <p>Compound: {result.vader.compound}</p>
              </div>

              {/* Hugging Face */}
              <div className="bg-white p-6 border rounded">
                <h3 className="text-purple-700 font-semibold mb-2">Hugging Face Results</h3>
                <p><strong>Sentiment:</strong> {result.huggingface.sentiment}</p>
                <p><strong>Polarity:</strong> {Number(result.huggingface.polarity).toFixed(2)}</p>
                <p className="italic mt-2"><strong>Generated Summary:</strong><br />
                  {result.huggingface.response.split('.').slice(0, 2).join('. ') + '.'}
                </p>
                <p className="mt-2 text-sm text-gray-600">
                  <strong>Note:</strong> The model analyzes the tone based on overall context and phrasing.
                  A polarity above 0.1 generally indicates a positive tone.
                </p>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
