export function formatDateET(timestampInSeconds) {
  return new Date(timestampInSeconds * 1000).toLocaleString('en-US', {
    timeZone: 'America/New_York',  // Eastern Time
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,  // use AM/PM format
    timeZoneName: 'short'
  });
}