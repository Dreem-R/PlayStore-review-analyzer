document.getElementById("trackBtn").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    function: trackCart,
    args: [
      document.getElementById("name").value,
      document.getElementById("email").value,
      document.getElementById("phone").value
    ]
  });
});

function trackCart(name, email, phone) {
  chrome.runtime.sendMessage({ action: "collect_cart", name, email, phone });
}