package net.xinerz.iulostandfound;

import android.content.Context;
import android.os.AsyncTask;
import android.util.Log;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import java.util.ArrayList;

public class NetworkActivity extends AsyncTask<String, Void, String> {
    HTTPParser jp = new HTTPParser();
    JSONObject objects = new JSONObject();
    JSONObject jo = new JSONObject();
    boolean found = true;
    String[] locations = {"Luddy", "Ballentine"};
    @Override
    protected String doInBackground(String[] params) {
        getJSON();
        //postJSON();
        return "some message";
    }

    @Override
    protected void onPostExecute(String message) {
        //process message
    }

    protected void getJSON(){
//        try{
//            jo = (JSONObject)jp.makeHttpRequest("http://149.161.157.123/data.json", "GET", objects).get("LostFind");
//
//            String flag = (String)jo.get("flag");
//
//            JSONObject jsonLost = (JSONObject) jo.get("lost");
//            String name = (String) jsonLost.get("name");
//            String lostUrl = (String) jsonLost.get("url");
//
//            JSONObject jsonResult = (JSONObject) jo.get("result");
//            String found = (String) jsonResult.get("found");
//            String foundUrl = (String) jsonResult.get("tag");
//
//            JSONArray jsonDB = (JSONArray) jo.get("database");
//            ArrayList<JSONObject> allObjects = new ArrayList<>();
//            for(int i = 0, size = jsonDB.length(); i < size; i++){
//                JSONObject foundObject = jsonDB.getJSONObject(i);
//                allObjects.add(foundObject);
//            }
//
//            Log.d("Get Json flag message", flag);
//            Log.d("Get Json lost name", name);
//            Log.d("Get Json lostUrl", lostUrl);
//            Log.d("Get Json foundUrl", foundUrl);
//            Log.d("array size", String.valueOf(allObjects.size()));
//
//        } catch (JSONException e) {
//            e.printStackTrace();
//        }
        Log.d("array size", "Hello HA!");
    }

    protected void postJSON(){
//        try {
//            JSONObject newObject = new JSONObject();
//            newObject.put("name", "book");
//            newObject.put("location", "Ballentine");
//            newObject.put("url", "./book.jpg");
//            ((JSONArray) jo.get("database")).put(newObject);
//            JSONArray jsonDB = (JSONArray) jo.get("database");
//
//            test
//            ArrayList<JSONObject> allObjects = new ArrayList<>();
//            for(int i = 0, size = jsonDB.length(); i < size; i++){
//                JSONObject foundObject = jsonDB.getJSONObject(i);
//                allObjects.add(foundObject);
//            }
//            Log.d("array size", String.valueOf(allObjects.size()));
//
//            jp.makeHttpRequest("http://149.161.157.123/data.json", "POST", newObject);
//        }catch (JSONException e) {
//            e.printStackTrace();
//        }
    }



}