param([string]$OutputDirectory = "build/speaker-sample")
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$speakerDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
$speakerSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speakerFormat = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$speakerTurns = @(
    @{speaker="A"; voice="Microsoft David Desktop"; text="Good morning. Today we are testing whether the transcription application can distinguish two speakers. I will speak first, and then my colleague will respond."},
    @{speaker="B"; voice="Microsoft Zira Desktop"; text="Thank you. I am the second speaker in this recording. We are taking turns so that each voice can be identified clearly, without anyone speaking at the same time."},
    @{speaker="A"; voice="Microsoft David Desktop"; text="This is the first speaker again. My label should stay the same even after another person has spoken. The weather is pleasant today, and we plan to meet again tomorrow."},
    @{speaker="B"; voice="Microsoft Zira Desktop"; text="And this is the second speaker returning. My label should also remain consistent. This short synthetic recording checks the integration, but real conversations will need separate testing."}
)
try {
    for ($speakerIndex = 0; $speakerIndex -lt $speakerTurns.Count; $speakerIndex++) {
        $speakerTurn = $speakerTurns[$speakerIndex]
        $speakerTurn.file = "turn-$speakerIndex.wav"
        $speakerSynth.SelectVoice($speakerTurn.voice)
        $speakerSynth.SetOutputToWaveFile((Join-Path $speakerDirectory $speakerTurn.file), $speakerFormat)
        $speakerSynth.Speak($speakerTurn.text)
        $speakerSynth.SetOutputToNull()
    }
    $speakerTurns | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $speakerDirectory "parts.json") -Encoding utf8
    Write-Output "Created four alternating synthetic speaker turns in $speakerDirectory"
} finally {
    $speakerSynth.Dispose()
}
