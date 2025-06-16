chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === "collect_cart") {
    chrome.tabs.sendMessage(sender.tab.id, {
      action: "scrape_cart",
      user: { name: msg.name, email: msg.email, phone: msg.phone }
    });
  }
});