# bootstrap — User Guide

## Welcome

**bootstrap** is a simple web app with a built-in assistant. From the main page, you can see the current app version and open the chat panel to ask questions.

This guide describes the app from the **user's point of view**.

---

## What you can do here

On the main page, you can:

- view the current app version
- view the deploy date shown on the page
- open the assistant panel
- ask questions in natural language
- read responses as they stream into the chat

---

## Using the chat panel

The chat panel opens from the button in the bottom-right corner of the page.

Inside the panel, you can:

- type a message and press **Enter** to send it
- press **Shift+Enter** to start a new line
- resize the panel by dragging its left edge
- switch between **user mode** and **dev mode**
- close the panel with the **×** button
- clear the conversation with the **trash** button

Assistant responses support markdown formatting, so replies may include:

- paragraphs
- bullet lists
- numbered lists
- links
- code blocks

---

## Conversation behavior

- The assistant replies directly in the chat panel.
- Responses stream into the UI instead of appearing all at once.
- Changing the mode starts a fresh conversation in the UI.
- Clearing the chat removes the current conversation from the UI and starts a fresh session.

---

## User mode

In **user mode**, the assistant focuses on helping you use the app.

You can ask it things like:

- what this app does
- what information is shown on the page
- how to use the chat panel
- what a specific control or button does
- how the current session behaves

---

## Dev mode

In **dev mode**, the assistant can answer more technical questions about how the app works.

This mode is intended for implementation-oriented or debugging-oriented questions.

---

## What appears on the page

The home page is intentionally minimal. You should expect to see:

- the app name
- the current version
- the deploy date
- the chat launcher button

If the assistant panel is open, you will also see:

- previous messages in the conversation
- the message input box
- the send button
- controls for mode switching, closing, or clearing the panel

---

## If something seems wrong

Try the following:

- refresh the page
- close and reopen the chat panel
- clear the conversation and send a new message
- check whether the page loaded fully before sending a message

If the app is reachable but the assistant does not respond, there may be a temporary backend issue.
